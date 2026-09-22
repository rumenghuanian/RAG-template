"""Agent 主循环：ReAct 风格的工具调用循环。

为什么手写而不用 LangChain 的 AgentExecutor：
1. 循环、终止条件、引用收集是这个项目的核心，必须能逐行讲清楚
2. LLM 可注入，没有 API key 也能用假 LLM 把循环逻辑测完整
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .identity import DEFAULT_IDENTITY, resolve_identity  # noqa: F401  （对外仍是这两个名字）

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """你是「{role}」，回答依据是一个{corpus}。

可用工具：
- search_notes(query, top_k)：混合检索，返回内容片段和 {id_name}
- read_article({id_name})：读某一篇全文
- list_articles({browse_arg})：列出清单{browse_desc}

工作方式：
1. 事实性问题**必须先检索**，不要凭记忆回答
2. 结果不相关或不够，就换关键词再检索；拿不准范围时先用 list_articles 看清单
3. 需要核对细节时才用 read_article，不要为了「看一眼」就整篇读进来
4. 只根据检索到的内容作答
5. 检索不到就明确说「{empty_phrase}」，不要编造
6. 不要断言工具没返回过的信息——比如某篇文章用什么语言写的、提到了什么，
   工具只给了标题时就说只看到标题

回答方式（很重要）：
- **直接回答问题，不要把整篇文章复述一遍。** read_article 是给你核对细节用的素材，不是让你转述给用户
- 默认先给结论（1~2 句），再给 3~5 条要点，全文控制在 400 字以内
- 用户明确说「展开讲 / 完整步骤 / 详细一点」时才展开
- 想深入的部分，告诉用户去看哪一篇（给出 {id_name}）比大段抄原文更有用
- 答案里只写出**你真正用到**的 {id_name}，格式如 `{id_hint}`；没用到的不要列

回答用中文，结构清晰。"""

# 领域相关的措辞全部走占位符（默认值在 agent/identity.py，全中性）：
# 这些词以前写死在 prompt 里，换个语料（菜谱）模型就会被告知"这是课程笔记库"、
# 示例 id 也还是 `week5/33…`，于是它会照着编造不存在的笔记 id。


def build_system_prompt(skill=None, identity: Optional[dict] = None) -> str:
    """用领域身份拼系统提示。

    `identity` 传进来时优先用它 —— `build_tools()` 会解析出一份**带真实 id_hint** 的身份，
    `RAGAgent` 把它透传过来，这样示例 id 一定来自语料本身。
    """
    merged = dict(identity) if identity else resolve_identity(skill)
    return SYSTEM_PROMPT_TEMPLATE.format(**merged)


# 向后兼容：老代码直接 import SYSTEM_PROMPT 时拿到的是通用版
SYSTEM_PROMPT = build_system_prompt()

# 累积「模型读到的原文」时的兜底上限（字符）：多步循环别让它无限涨
_MAX_CONTEXT_CHARS = 40000

FORCE_ANSWER = (
    "已达到检索步数上限。请只根据上面已经拿到的信息作答；"
    "如果仍然不足以回答，就直接说明信息不足。"
)


@dataclass
class AgentResult:
    answer: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "answered"
    needs_human: bool = False
    # **模型这一轮真正读到的原文**（每条 = 一个内容块）。
    # 为什么必须留：`citations` 只有 article_id/title，没有正文；而判「答案有没有编造」
    # 必须拿模型实际看到的上下文当参照 —— 写评测时这个字段缺过一次，
    # 导致只能拿 gold 当参照，把「引用了上下文里其他文章」误判成编造。
    contexts: List[Dict[str, Any]] = field(default_factory=list)

    def sources_text(self) -> str:
        if not self.citations:
            return ""
        return "\n".join(
            f"- {c.get('title')}（{c.get('article_id')}）" for c in self.citations
        )


class RAGAgent:
    def __init__(
        self,
        llm,
        tools,
        system_prompt: Optional[str] = None,
        max_steps: int = 6,
    ):
        self.llm = llm
        self.tools = tools
        # 领域身份从工具的 registry 上取（build_tools 挂着解析好的 identity + skill），
        # 这样换领域**不需要改任何构造点**：prompt 与工具描述会自己跟着 skill 走。
        # identity 里带**真实 id_hint**，优先用它（见 agent/identity.py）。
        self.system_prompt = system_prompt or build_system_prompt(
            getattr(tools, "skill", None), getattr(tools, "identity", None)
        )
        self.max_steps = max_steps

    # ---------- 主循环 ----------
    def run(
        self, question: str, history: Optional[List[Dict[str, Any]]] = None
    ) -> AgentResult:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        messages.extend(history or [])
        messages.append({"role": "user", "content": question})

        citations: List[Dict[str, Any]] = []
        steps: List[Dict[str, Any]] = []
        contexts: List[Dict[str, Any]] = []

        for step in range(1, self.max_steps + 1):
            reply = self.llm.chat(messages, tools=self.tools.schemas())

            if not reply.wants_tool:
                answer = (reply.content or "").strip()
                if not answer:
                    return self._finish(
                        "模型没有返回内容。", citations, steps, "empty_reply", contexts
                    )
                return self._finish(answer, citations, steps, "answered", contexts)

            messages.append(_assistant_message(reply))
            for call in reply.tool_calls:
                result = self.tools.invoke(call.name, call.arguments)
                ok = "error" not in result
                logger.info(
                    "步骤 %d：%s(%s) -> %s",
                    step,
                    call.name,
                    json.dumps(call.arguments, ensure_ascii=False),
                    "ok" if ok else result["error"],
                )
                steps.append(
                    {
                        "step": step,
                        "tool": call.name,
                        "arguments": call.arguments,
                        "ok": ok,
                        "error": result.get("error"),
                        # 语义检索被相关性闸门拦下（一条都没返回）
                        "empty": call.name == "search_notes"
                        and not result.get("results"),
                    }
                )
                citations.extend(_citations_from(result))
                # 兜底上限：多步循环时别让上下文无限涨（灌给裁判也没意义）
                if sum(len(c["text"]) for c in contexts) < _MAX_CONTEXT_CHARS:
                    contexts.extend(_contexts_from(result, call.name))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id or call.name,
                        "name": call.name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        # 步数用尽：不带工具再问一次，逼它给结论
        logger.warning("达到最大步数 %d，强制收尾", self.max_steps)
        messages.append({"role": "user", "content": FORCE_ANSWER})
        reply = self.llm.chat(messages)
        answer = (reply.content or "").strip() or "信息不足，无法回答。"
        return self._finish(answer, citations, steps, "max_steps", contexts)

    # ---------- 辅助 ----------
    @staticmethod
    def _finish(
        answer: str,
        citations: List[Dict[str, Any]],
        steps: List[Dict[str, Any]],
        stop_reason: str,
        contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentResult:
        unique = _dedup_citations(citations)
        # 检索到的 ≠ 用到的。不过滤的话出处列表会虚高：
        # 实测一次回答用了 2 篇，却列了 6 篇（另外 4 篇只是检索命中）。
        used = [c for c in unique if _mentioned(c.get("article_id"), answer)]
        if used:
            unique = used
        elif unique:
            logger.warning("答案里没有引用任何 article_id，保留全部检索结果作出处")
        # 调了工具却零出处，或者每次语义检索都被相关性闸门拦下
        # → 内容层面没有依据，交给人。
        # 只看出处不够：list_articles 返回的是目录条目，模型引用它并不等于有内容依据
        # （实测「Rust 写内核」那轮就是这样被误判成「有依据」的）。
        searches = [s for s in steps if s.get("tool") == "search_notes"]
        all_gated = bool(searches) and all(s.get("empty") for s in searches)
        needs_human = bool(steps) and (not unique or all_gated)
        return AgentResult(
            answer=answer,
            citations=unique,
            steps=steps,
            stop_reason=stop_reason,
            needs_human=needs_human,
            contexts=contexts or [],
        )


def _contexts_from(result: Dict[str, Any], tool: str = "") -> List[Dict[str, Any]]:
    """把一次工具返回里**有正文的部分**抽成内容块，供评测拿真实上下文当参照。

    只收「模型能读到的文字」：
      - `read_article` 的 `content`（全文）
      - `search_notes` 的 `results[].snippet`（片段）
    **刻意不收 `list_articles` 的目录条目** —— 那只有标题，没有内容；
    把它当上下文会虚增「有依据」（这个坑在 `needs_human` 的判定里踩过一次）。
    """
    out: List[Dict[str, Any]] = []
    if result.get("article_id") and result.get("content"):
        out.append(
            {
                "tool": tool,
                "article_id": result.get("article_id"),
                "title": result.get("title"),
                "text": result["content"],
            }
        )
    for item in result.get("results") or []:
        if item.get("snippet"):
            out.append(
                {
                    "tool": tool,
                    "article_id": item.get("article_id"),
                    "title": item.get("title"),
                    "text": item["snippet"],
                }
            )
    return out


def _mentioned(article_id: Optional[str], answer: str) -> bool:
    """答案里有没有真的引用这篇（允许只写去掉 .md 的 id）。"""
    if not article_id:
        return False
    if article_id in answer:
        return True
    stem = article_id[:-3] if article_id.endswith(".md") else article_id
    return stem in answer


def _assistant_message(reply) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": reply.content or "",
        "tool_calls": [
            {
                "id": c.id or c.name,
                "type": "function",
                "function": {
                    "name": c.name,
                    "arguments": json.dumps(c.arguments, ensure_ascii=False),
                },
            }
            for c in reply.tool_calls
        ],
    }


def _citations_from(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    if result.get("article_id"):  # read_article 直接返回单篇
        return [_pick(result)]
    items = result.get("results") or result.get("articles") or []
    return [_pick(it) for it in items if it.get("article_id")]


def _pick(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "article_id": item.get("article_id"),
        "title": item.get("title"),
        "week": item.get("week"),
    }


def _dedup_citations(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for c in citations:
        key = c.get("article_id")
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
