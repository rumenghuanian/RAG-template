"""Agent 层的「零成本」对照评测：不调用任何 LLM，用脚本策略驱动真实的 RAGAgent。

为什么能零成本
--------------
`RAGAgent(llm, tools, ...)` 的 llm 是注入进来的，只要实现一个 `chat(messages, tools)`
就能跑完整循环；而本文件关心的指标全部落在**检索层**（这一轮到底碰到过哪些文章），
不关心答案文字，所以既不花钱也不需要人工评分。

这个脚本想回答的问题
--------------------
Agent 相对单轮 RAG 的收益，只有两个可能来源：

  ① **会重试**：第一次检索被相关性闸门拦下 / 没命中，换个说法再来一次；
  ② **看得更多**：多轮检索的并集，本身就比一次取 5 篇大。

② 完全不需要 Agent —— 把 `top_k` 调大就行。所以真正的对照是**等预算**的：

  single5        单轮，取 5 篇（= 现有基线，adaptive 82.9%）
  single25       单轮，取 25 篇 —— 和 agent 5 次 × 5 篇一个量级，无循环
  agent_stop     脚本 agent，一有结果就停（只衡量 ①）
  agent_explore  脚本 agent，把改写变体全跑完（衡量 ① + ②）

如果 agent_explore ≈ single25 → 多查询不比直接多取划算，**不必为 agent 循环付费**，
调大 top_k 即可；如果 agent_explore 明显 > single25 → 「改写」本身有价值，
才值得花钱请 LLM 来改写（第 1 档评测）。

**诚实边界**：脚本策略是我写的规则，它不判断「结果对不对」，也看不到 gold。
所以 agent 两臂的数字是**这套机制在给定策略下的上限**，不等于 DeepSeek 的真实水平。
它唯一的用途是：**先证伪**。上限都不比单轮高，就没必要花真钱。

用法：
    python eval/run_agent_eval.py
    python eval/run_agent_eval.py --json
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 让 scoring 可导入
os.chdir(ROOT)

import bootstrap  # noqa: E402

bootstrap.prepare()

from dotenv import load_dotenv  # noqa: E402

load_dotenv(bootstrap.env_file_for(os.getenv("EVAL_SKILL", "notes")))

from agent import RAGAgent, build_tools  # noqa: E402
from agent.llm import LLMReply, ToolCall  # noqa: E402
from rag_core import BasicRAGPipeline, RAGConfig  # noqa: E402
from rag_core.skills import SKILLS  # noqa: E402
import scoring  # noqa: E402

TOP_N = 5                 # agent 单次检索取几篇（与生产 tools 默认一致）
EQUAL_BUDGET = TOP_N * 5  # 等预算单轮臂取几篇
# 类别口径只有一份，在 scoring 里（这里再抄一份就漏修过一次，见 scoring.bucket_of）
ANSWERABLE = list(scoring.ANSWERABLE_KINDS)
# 不参与 hit@5：absent = 术语完全不出现；mentioned = 术语出现但语料答不了
GATE_KINDS = list(scoring.GATE_KINDS)
MANUAL_KINDS = list(scoring.MANUAL_KINDS)

# ---------- 查询改写的规则（只吃问题文本，不看 gold） ----------
_PUNCT = re.compile(r"[，。？！、；：\"'“”‘’（）()《》\[\]【】{}!?,.;:\s]+")
# 长词排前面，否则「有哪些」会被「哪些」先咬掉一半
_INTERROG = sorted(
    [
        "什么是", "是什么", "为什么", "为什么说", "怎么", "怎样", "如何", "为啥",
        "哪些", "哪个", "哪一", "哪一种", "是否", "有没有", "介绍", "说明", "解释",
        "一下", "分别", "以及", "有哪些", "这么", "那么", "到底",
        "吗", "呢", "的", "了", "请", "我", "要", "会",
    ],
    key=len,
    reverse=True,
)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]{1,}")


def _strip_question(q: str) -> str:
    """去掉疑问词和标点，留下内容词。"""
    out = _PUNCT.sub(" ", q)
    for w in _INTERROG:
        out = out.replace(w, " ")
    return re.sub(r"\s+", " ", out).strip()


def _longest_clause(q: str) -> str:
    """取最长的一个分句 —— 长句里通常只有一半是检索线索。"""
    parts = [p.strip() for p in _PUNCT.split(q) if p.strip()]
    return max(parts, key=len) if parts else q


def build_ladder(q: str) -> list:
    """同一问题的多种检索说法，按「先原句、后改写」排序，去重。"""
    idents = " ".join(_IDENT.findall(q))
    cands = [q, _strip_question(q), idents, _longest_clause(q)]
    out, seen = [], set()
    for c in cands:
        c = (c or "").strip()
        if len(c) < 2 or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _ids(result: dict) -> list:
    """从**内容检索**的返回值里取出这次碰到了哪些文章。

    只看 search_notes / read_article，**刻意不看 list_articles**：
    目录清单只是标题，看到标题不等于看到内容。把它算进覆盖率会让 agent
    每条都「命中」——这正是 needs_human 判断里踩过的同一个坑。
    """
    if result.get("article_id"):  # read_article
        return [result["article_id"]]
    return [it.get("article_id") for it in (result.get("results") or []) if it.get("article_id")]


class RecordingTools:
    """包一层真实 ToolRegistry：生产行为一字不改，只额外记录碰到过哪些文章。

    必须走真实 registry，因为相关性闸门（min_relevance）就在 search_notes 里面，
    自己模拟一套闸门就等于在测另一个系统。
    """

    def __init__(self, inner, sink: set):
        self.inner = inner
        self.sink = sink

    def names(self):
        return self.inner.names()

    def schemas(self):
        return self.inner.schemas()

    def invoke(self, name, arguments):
        result = self.inner.invoke(name, arguments)
        self.sink.update(_ids(result))
        return result


class ScriptedLLM:
    """脚本化「agent 策略」：只看问题文本和工具返回，绝不看 gold。

    policy="stop"    一次检索有结果就停 —— 衡量「闸门触发重试」的收益
    policy="explore" 把改写变体跑完再停 —— 衡量「多查询并集」的收益上限
    """

    def __init__(self, question: str, policy: str = "stop", top_k: int = TOP_N):
        self.ladder = build_ladder(question)
        self.policy = policy
        self.top_k = top_k
        self.read_top = True

    # ---- 内部 ----
    def _next_query(self):
        return self.ladder.pop(0) if self.ladder else None

    @staticmethod
    def _count(messages, name):
        return sum(1 for m in messages if m.get("role") == "tool" and m.get("name") == name)

    def _search(self, query):
        return LLMReply(
            tool_calls=[
                ToolCall(name="search_notes", arguments={"query": query, "top_k": self.top_k}, id="c1")
            ]
        )

    def _read(self, article_id):
        return LLMReply(
            tool_calls=[ToolCall(name="read_article", arguments={"article_id": article_id}, id="c2")]
        )

    @staticmethod
    def _done():
        # 内容不重要：评测只看 steps 里的文章，不看答案文字
        return LLMReply(content="[脚本策略收尾]")

    # ---- 主决策 ----
    def chat(self, messages, tools=None):
        if not tools:  # FORCE_ANSWER 那一轮，循环已经要收尾了
            return self._done()

        last = messages[-1] if messages else {}
        if last.get("role") != "tool":  # 第一步
            q = self._next_query()
            return self._search(q) if q else self._done()

        name, payload = last.get("name"), last.get("content") or "{}"
        try:
            result = json.loads(payload)
        except json.JSONDecodeError:
            result = {}

        if name == "search_notes":
            if result.get("results"):
                q = self._next_query() if self.policy == "explore" else None
                if q:
                    return self._search(q)
                top = result["results"][0]
                if self.read_top and not self._count(messages, "read_article"):
                    return self._read(top.get("article_id"))
                return self._done()
            # 空结果（被闸门拦下）→ 换说法
            q = self._next_query()
            if q:
                return self._search(q)
            if not self._count(messages, "list_articles"):
                return LLMReply(tool_calls=[ToolCall(name="list_articles", arguments={}, id="c3")])
            return self._done()

        if name == "read_article":
            q = self._next_query() if self.policy == "explore" else None
            return self._search(q) if q else self._done()

        return self._done()  # list_articles 之后收尾


def run_arm(question, pipeline, policy, min_relevance):
    """跑一次脚本 agent，返回 (碰到的文章集合, 步骤数, needs_human, 检索全被拦)。"""
    touched: set = set()
    tools = RecordingTools(build_tools(pipeline, min_relevance=min_relevance), touched)
    agent = RAGAgent(ScriptedLLM(question, policy=policy), tools, max_steps=6)
    result = agent.run(question)
    searches = [s for s in result.steps if s.get("tool") == "search_notes"]
    all_gated = bool(searches) and all(s.get("empty") for s in searches)
    return touched, len(result.steps), result.needs_human, all_gated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="写明细到 eval/last_agent_report.json")
    args = parser.parse_args()

    skill_name = os.getenv("EVAL_SKILL", "notes")
    golden = [
        json.loads(line)
        for line in Path(bootstrap.artifact_path("golden", skill_name, ".jsonl", str(ROOT / "eval"))).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    config = RAGConfig()
    config.validate()
    pipeline = BasicRAGPipeline(config, SKILLS[skill_name]()).build()

    arms = ["single5", "single10", "single25", "agent_stop", "agent_explore"]
    per_kind = {
        k: {"n": 0, "hit": {a: 0 for a in arms}, "cover": {a: 0.0 for a in arms}}
        for k in ANSWERABLE
    }
    steps_sum, seen_sum = 0.0, 0.0
    n_answerable = 0
    gate_items, detail = [], []
    rank_buckets = {"≤5": 0, "≤10": 0, "≤25": 0, "≤99": 0, "未召回": 0}

    for item in golden:
        kind, q, expect = item["kind"], item["question"], set(item["expect"])

        full = [d.metadata.get("article_id") for d in pipeline.retrieve(q, top_k=10 ** 6)]
        got = {
            "single5": set(full[:TOP_N]),
            "single10": set(full[:10]),
            "single25": set(full[:EQUAL_BUDGET]),
        }
        # 诊断：gold 在单轮结果里排第几 —— 排名问题还是召回问题，一眼能分
        gold_rank = None
        if kind not in GATE_KINDS and expect:
            rank = next((i + 1 for i, a in enumerate(full) if a in expect), None)
            gold_rank = rank
            if rank is None:
                rank_buckets["未召回"] += 1
            else:
                key = "≤5" if rank <= 5 else "≤10" if rank <= 10 else "≤25" if rank <= 25 else "≤99"
                rank_buckets[key] += 1
        meta = {}
        for arm, policy in (("agent_stop", "stop"), ("agent_explore", "explore")):
            touched, steps, needs_human, all_gated = run_arm(
                q, pipeline, policy, config.min_relevance
            )
            got[arm] = touched
            meta[arm] = {
                "steps": steps,
                "touched": len(touched),
                "needs_human": needs_human,
                "all_gated": all_gated,
            }

        if kind in GATE_KINDS:
            gate_items.append({"kind": kind, "q": q, **meta})
        elif kind in MANUAL_KINDS:
            # partial 不进命中率口径：它的 expect 是空的（判据是「有没有明说不完全」，
            # 在生成层用 cited_any 量）。硬算只会得到恒 0 的假数字，而且**永远不动**，
            # 在门禁里是一行废行 —— 之前报告里 partial 五个臂全是 0.0% 就是这么来的。
            # 这里仍然记进 detail（下面的 steps/seen 均值也不含它，口径和上面的表一致）。
            pass
        else:
            b = per_kind.setdefault(
                kind, {"n": 0, "hit": {a: 0 for a in arms}, "cover": {a: 0.0 for a in arms}}
            )
            b["n"] += 1
            n_answerable += 1
            steps_sum += meta["agent_explore"]["steps"]
            seen_sum += meta["agent_explore"]["touched"]
            for a in arms:
                if expect & got[a]:
                    b["hit"][a] += 1
                if expect:
                    b["cover"][a] += len(expect & got[a]) / len(expect)

        detail.append(
            {
                "kind": kind,
                "question": q,
                "expect": sorted(expect),
                "gold_rank": gold_rank,
                "got": {a: sorted(x for x in got[a] if x) for a in arms},
                "agent": meta,
            }
        )

    total = sum(v["n"] for k, v in per_kind.items() if k in ANSWERABLE)
    print(f"\nskill={skill_name}  标注 {len(golden)} 条，其中可答 {total} 条；闸门类 {len(gate_items)} 条")
    print("不调用任何 LLM；agent 两臂由脚本策略驱动，衡量的是机制上限。\n")

    head = f"{'类型':<12}{'n':>3}"
    for a in arms:
        head += f"{a:>14}"
    print(head)
    print("-" * len(head))
    for kind in ANSWERABLE:
        b = per_kind.get(kind)
        if not b or not b["n"]:
            continue
        row = f"{kind:<12}{b['n']:>3}"
        for a in arms:
            row += f"{b['hit'][a] / b['n']:>13.1%} "
        print(row)

    print(f"\n{'总体':<12}{total:>3}", end="")
    for a in arms:
        hit = sum(per_kind[k]["hit"][a] for k in ANSWERABLE) / total
        print(f"{hit:>13.1%} ", end="")
    print("   hit@seen（碰到的文章里有没有 gold）")

    print(f"\n{'总体':<12}{total:>3}", end="")
    for a in arms:
        cov = sum(per_kind[k]["cover"][a] for k in ANSWERABLE) / total
        print(f"{cov:>13.1%} ", end="")
    print("   cover（multi 类才有意义）")

    print("\n瓶颈定位：gold 在单轮检索结果里的最好名次（可答题目）")
    for key in ["≤5", "≤10", "≤25", "≤99", "未召回"]:
        n = rank_buckets[key]
        extra = "  ← single5 够得着" if key == "≤5" else ""
        print(f"  名次 {key:<7}{n:>3}/{total}{n / max(total, 1):>8.1%}{extra}")
    print("  （≤5 之外的数量大 → gold 被检索到了，只是排在 5 名之后：**排序**问题，")
    print("    该上重排序器，而不是让 agent 多搜几轮）")

    print("\n关键对照：多查询并集 vs 等预算单轮")
    for kind in ANSWERABLE + ["总体"]:
        if kind == "总体":
            d = sum(per_kind[k]["hit"]["agent_explore"] - per_kind[k]["hit"]["single25"] for k in ANSWERABLE) / total
        else:
            b = per_kind.get(kind)
            if not b or not b["n"]:
                continue
            d = (b["hit"]["agent_explore"] - b["hit"]["single25"]) / b["n"]
        print(f"  {kind:<12}agent_explore - single25 = {d:>+7.1%}")

    print("\n「只会重试」够不够：agent_stop vs single5")
    for kind in ANSWERABLE:
        b = per_kind.get(kind)
        if not b or not b["n"]:
            continue
        d = (b["hit"]["agent_stop"] - b["hit"]["single5"]) / b["n"]
        print(f"  {kind:<12}{d:>+7.1%}")

    print("\n代价（可答题目平均）：")
    print(f"  agent_explore 步骤数 {steps_sum / max(n_answerable, 1):.1f}，"
          f"实际看过文章 {seen_sum / max(n_answerable, 1):.1f} 篇 "
          f"（single25 预算 {EQUAL_BUDGET} 篇，agent_stop 预算 {TOP_N} 篇）")

    if gate_items:
        print(f"\n闸门类 {len(gate_items)} 条（Agent 该不该转人工）：")
        for kind in GATE_KINDS:
            group = [r for r in gate_items if r["kind"] == kind]
            if not group:
                continue
            stay = sum(1 for r in group if r["agent_explore"]["needs_human"])
            gated = sum(1 for r in group if r["agent_explore"]["all_gated"])
            print(f"  {kind:<10} n={len(group)}  needs_human {stay}/{len(group)}"
                  f"  所有检索被闸门拦下 {gated}/{len(group)}")
        for r in gate_items:
            m = r["agent_explore"]
            print(f"  [{r['kind']:<9}] steps={m['steps']} touched={m['touched']} "
                  f"{'转人工' if m['needs_human'] else '未转人工'}  {r['q']}")

    if args.json:
        out = Path(bootstrap.artifact_path("last_agent_report", skill_name, ".json", str(ROOT / "eval")))
        out.write_text(
            json.dumps(
                {
                    "skill": skill_name,
                    "arms": arms,
                    "top_n": TOP_N,
                    "equal_budget": EQUAL_BUDGET,
                    "per_kind": {
                        k: {
                            "n": v["n"],
                            "hit_rate": {a: v["hit"][a] / v["n"] for a in arms} if v["n"] else {},
                            "cover": {a: v["cover"][a] / v["n"] for a in arms} if v["n"] else {},
                        }
                        for k, v in per_kind.items()
                    },
                    "gate_items": gate_items,
                    "detail": detail,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n明细已写到 {out}")


if __name__ == "__main__":
    main()
