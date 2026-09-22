"""Agent 工具：内容检索、读全文、列清单。

工具是 Agent 和 RAG 检索层之间唯一的接口面。
`build_tools(index)` 只要求 index 提供两样东西（鸭子类型，方便测试替换）：
  - `.documents`：父文档列表（每篇带 article_id / title / week 元数据）
  - `.retrieve(question, top_k)`：返回命中的子块
"""
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 600
MAX_ARTICLE_CHARS = 12000


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., Dict[str, Any]]

    def schema(self) -> Dict[str, Any]:
        """OpenAI function-calling 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: List[Tool], skill=None):
        self._tools = {t.name: t for t in tools}
        # 领域身份挂在这里，`RAGAgent` 会自动读取它来拼 SYSTEM_PROMPT ——
        # 这样换领域不用改任何构造点
        self.skill = skill

    def names(self) -> List[str]:
        return list(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def invoke(self, name: str, arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """永远返回 dict，出错也返回 dict —— 让模型看到错误并自己纠正。"""
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"没有这个工具：{name}。可用工具：{self.names()}"}
        try:
            return tool.func(**(arguments or {}))
        except TypeError as e:
            return {"error": f"参数不对：{e}"}
        except Exception as e:  # noqa: BLE001 - 工具异常一律交给模型处理
            logger.warning("工具 %s 执行失败: %s", name, e)
            return {"error": f"{type(e).__name__}: {e}"}


def _citation(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "article_id": meta.get("article_id"),
        "title": meta.get("title"),
        "week": meta.get("week"),
    }


def build_tools(index, min_relevance: float = 0.65) -> ToolRegistry:
    documents = list(index.documents)
    by_id = {d.metadata.get("article_id"): d for d in documents}

    def search_notes(query: str, top_k: int = 5) -> Dict[str, Any]:
        hits = index.retrieve(query, top_k=int(top_k))
        if not hits:
            return {
                "results": [],
                "note": "没有命中任何文章。换个说法，或先用 list_articles 看看有哪些文章。",
            }
        # 相关性闸门：检索永远会返回 K 条，RRF 分数又没有区分度，
        # 只能靠余弦相似度判断「语料里到底有没有这件事」。
        best = max((d.metadata.get("vector_cos") or 0.0) for d in hits)
        if best < min_relevance:
            return {
                "results": [],
                "max_similarity": round(best, 3),
                "note": (
                    f"语义检索没找到足够相关的内容（最相似 {best:.2f}，阈值 {min_relevance:.2f}）。"
                    "有两种可能：① 笔记里确实没有这个主题；"
                    "② 这个问题不适合语义检索，比如「第几周讲了什么」这类结构性提问 "
                    "（实测它的相似度甚至比语料外的问题还低）。"
                    "请先换个更具体的关键词重试，或用 list_articles 看文章清单；"
                    "两者都没有结果，就直接告诉用户笔记里没有，不要编造。"
                ),
            }
        results = [
            {**_citation(d.metadata), "snippet": d.page_content[:SNIPPET_CHARS]}
            for d in hits
        ]
        return {"results": results, "max_similarity": round(best, 3)}

    def read_article(article_id: str) -> Dict[str, Any]:
        doc = by_id.get(article_id)
        if doc is None:
            return {
                "error": f"没有这篇文章：{article_id}",
                "available_sample": [k for k in list(by_id)[:5]],
            }
        content = doc.page_content
        truncated = len(content) > MAX_ARTICLE_CHARS
        return {
            **_citation(doc.metadata),
            "content": content[:MAX_ARTICLE_CHARS],
            **({"note": "文章过长，已截断"} if truncated else {}),
        }

    def list_articles(**kwargs) -> Dict[str, Any]:
        # 参数名由 skill 决定（notes 是 week、recipe 是 category），所以这里吃 **kwargs。
        # 直接把 schema 里的名字改掉、函数还写 `week=None` 的话，
        # 模型按 schema 传 category 会 TypeError —— 自己引入的坑，别踩。
        scope = kwargs.get(browse_arg)
        items = []
        for d in sorted(
            documents,
            key=lambda x: (x.metadata.get("week") or 0, x.metadata.get("article_no") or 0),
        ):
            m = d.metadata
            if scope is not None and str(m.get(browse_field)) != str(scope):
                continue
            items.append(
                {
                    **_citation(m),
                    "article_no": m.get("article_no"),
                    "is_thinking": m.get("is_thinking"),
                }
            )
        return {"count": len(items), "articles": items}

    # 领域措辞全部来自 skill.agent_identity：写死会让模型在换领域后
    # 仍被告知"这是课程笔记库"、照着一个不存在的 id 格式去编（实测过）。
    identity = dict(getattr(getattr(index, "skill", None), "agent_identity", None) or {})
    corpus = identity.get("corpus", "知识库")
    id_name = identity.get("id_name", "article_id")
    id_hint = identity.get("id_hint", "week5/29.RAG是怎么工作的.md")
    browse_arg = identity.get("browse_arg", "week")
    browse_field = identity.get("browse_field", browse_arg)
    browse_desc = identity.get("browse_desc", "")

    return ToolRegistry(
        skill=getattr(index, "skill", None),
        tools=[
            Tool(
                name="search_notes",
                description=(
                    f"在{corpus}里做语义+关键词混合检索，返回最相关的内容片段和 {id_name}。"
                    "回答事实性问题前应该先用它。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "检索用的关键词或问题"},
                        "top_k": {
                            "type": "integer",
                            "description": "返回几篇（默认 5，最多 10）",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
                func=search_notes,
            ),
            Tool(
                name="read_article",
                description="读某一篇的全文。需要细节、步骤或完整论述时用。",
                parameters={
                    "type": "object",
                    "properties": {
                        id_name: {
                            "type": "string",
                            "description": f"内容 id，形如 {id_hint}",
                        }
                    },
                    "required": [id_name],
                },
                func=read_article,
            ),
            Tool(
                name="list_articles",
                description=f"列出{corpus}的清单{browse_desc}。适合回答「都有哪些」这类问题。",
                parameters={
                    "type": "object",
                    "properties": {
                        browse_arg: {
                            "type": "string",
                            "description": (
                                f"范围筛选{browse_desc}。不传则列出全部。"
                                if browse_desc
                                else "（本领域无筛选维度，不传即可）"
                            ),
                        }
                    },
                    "required": [],
                },
                func=list_articles,
            ),
        ],
    )
