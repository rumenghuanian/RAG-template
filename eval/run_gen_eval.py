"""生成层评测：答案正确性 / 引用准确性 / 编造检测。

为什么需要它
------------
`run_eval.py` 只测到「检索到了正确文章」就停了。但用户看到的是**答案**。
实测证明这个缺口是真的：`AgentSkillBot 是怎么实现的？` 这类题在检索层算 **hit**
（文章里确实有 `AgentSkillBot` 这个词），但那篇文章只把词用在代码示例的请求头里，
**根本没有回答「怎么实现」** —— 检索层的 hit@5 原理上测不出这件事。

两个臂
------
  single  单轮 RAG（`pipeline.query`）—— prompt 要求「标明出处（文章标题）」
  agent   ReAct Agent（`RAGAgent`）—— SYSTEM_PROMPT 要求引用 article_id

⚠ 两个臂的**引用格式不一致**（实测各 100% 用自己的那套），这本身就是个该修的产品问题。

指标（先分清哪些是确定性的，哪些要 LLM 裁判）
----------------------------------------------
确定性、零额外成本：
  context_ok     正确文章有没有进到生成上下文里（**这是检索的锅**）
  cited          答案正文里有没有真的点出那篇文章（id / id 去 .md / 标题 任一）
  cited_right    = context_ok and cited —— **材料给了且引对了**。这是主指标，
                 **不等于「答对了」**，内容正确性要 `--judge`。
  abstained      开头有没有声明「笔记里没有」（结构化判定，见 eval/scoring.py）
  pure_abstain   = abstained and not cited —— 既声明缺口又没指向任何语料内容，是真的没答

闸门类（语料答不了的问题）：
  declares_gap   = abstained —— 该说「笔记里没有」，**这是正确行为**
  silent_gap     = not abstained —— 没声明缺口就作答。**注意这不等于「编造」**：
                 诚实地说「没有专门章节，但散落的内容是这些」也是不声明却合理的。
                 要判真编造得看 `--judge`。

**先跑生成、再判分**：`--rescore` 用已保存的答案重算确定性指标，
`--judge-only` 只补裁判。改判分规则不必重新生成（省时省钱）。

用法：
    python eval/run_gen_eval.py --limit 4      # 小样本验证判分逻辑
    python eval/run_gen_eval.py --judge        # 全量 + 裁判
    python eval/run_gen_eval.py --rescore      # 用已保存答案重算指标（不调 LLM）
    python eval/run_gen_eval.py --judge-only   # 只补裁判
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 让 scoring 可导入
os.chdir(ROOT)

import bootstrap  # noqa: E402

bootstrap.prepare()

from dotenv import load_dotenv  # noqa: E402

load_dotenv(bootstrap.env_file_for(os.getenv("EVAL_SKILL", "notes")))

# 评测产物按 skill 分开：否则换语料跑一次，上一套报告/标注就被覆盖了
# （notes 沿用旧文件名，其它 skill 加后缀，见 bootstrap.artifact_path）
SKILL = os.getenv("EVAL_SKILL", "notes")
EVAL_DIR = str(ROOT / "eval")
REPORT = Path(bootstrap.artifact_path("last_gen_report", SKILL, ".json", EVAL_DIR))
SUBSET = Path(bootstrap.artifact_path("last_gen_subset", SKILL, ".json", EVAL_DIR))
# 对照臂单独存一份，**绝不覆盖真实报告** —— 它的裁判结果是对着错上下文打的
SHUFFLE = Path(bootstrap.artifact_path("last_gen_shuffle", SKILL, ".json", EVAL_DIR))
GOLDEN = Path(bootstrap.artifact_path("golden", SKILL, ".jsonl", EVAL_DIR))

from langchain_community.callbacks import get_openai_callback  # noqa: E402

from agent import OpenAICompatLLM, RAGAgent, build_tools  # noqa: E402
from rag_core import BasicRAGPipeline, RAGConfig  # noqa: E402
from rag_core.skills import SKILLS  # noqa: E402
from scoring import abstained as _abstained  # noqa: E402
from scoring import cited as _cited  # noqa: E402
import scoring  # noqa: E402
from scoring import cited_any as _cited_any  # noqa: E402
from scoring import key_point_coverage as _coverage  # noqa: E402

# 类别口径只有一份，在 scoring 里 —— 各脚本各抄一份的结果就是漏修（见 scoring.bucket_of）
ANSWERABLE = list(scoring.ANSWERABLE_KINDS)
GATE_KINDS = list(scoring.GATE_KINDS)
# 人工判定类：语料有相关内容但不完全回答。既不算可答（没有干净 gold），也不算闸门类。
MANUAL_KINDS = list(scoring.MANUAL_KINDS)


def _is_answerable(row) -> bool:
    return scoring.bucket_of(row["kind"]) == "answerable"


def _rotate(refs):
    """对照臂用：把参照物轮转一位，让每条答案拿到**别题**的上下文。

    用轮转而不是随机打乱 —— 小样本下随机可能把某条自己的上下文还给它（固定点），
    轮转**保证没有固定点**。这条性质有测试钉着：对照臂一旦有固定点，
    「忠实度没下降」就可能是那几条自己跟自己比出来的，结论作废。
    """
    return list(refs[1:] + refs[:1]) if len(refs) > 1 else list(refs)


def _sync_labels(detail, golden_path) -> int:
    """把报告里的答案按**当前** golden.jsonl 重新绑定 kind / expect。

    答案是一次性生成的（贵），标注是会改的（便宜）。修改标注后不该被迫重新生成，
    所以重算时按问题文本把标注同步过来。
    """
    golden = [
        json.loads(line)
        for line in Path(golden_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_q = {g["question"]: g for g in golden}
    changed = 0
    for row in detail:
        g = by_q.get(row["question"])
        if not g:
            continue
        if row.get("kind") != g["kind"] or row.get("expect") != g["expect"]:
            changed += 1
        row["kind"], row["expect"] = g["kind"], g["expect"]
    return changed

# ---------- 裁判的参照物上限：**这是整个裁判口径的关键参数** ----------
# 踩过的坑：`ref[:9000]` 之后 `_judge` 里又 `gold[:3000]`，实际只喂 3000 字。
# 实测 6 条 easy 的参照物是 5,378–24,671 字 → 裁判只看到 12%–56%，
# 于是**答案里凡是来自被截掉部分的内容全被判成编造**：真实上下文 0/6 忠实，
# 打乱对照也是 0/6 —— 指标钉死在 0，没有区分力。照规矩：先怀疑仪器。
JUDGE_REF_CHARS = 24000     # 参照物目标上限（按块比例分配，保证每篇都露头）
JUDGE_ANSWER_CHARS = 6000   # 答案上限（实测最长约 3,100 字，留一倍余量）
# 推理模型会先思考再输出：参照物一大，思考就长，2500 不够 24,000 字参照物用
# （实测 6 条里 2 条返回空 content → 判分失败）。这个数是量出来的，不是猜的。
JUDGE_MAX_TOKENS = int(os.getenv("JUDGE_MAX_TOKENS", "6000"))


def _fit_reference(blocks, cap: int = JUDGE_REF_CHARS) -> str:
    """把上下文块拼成裁判的参照物。

    **按长度比例**给每块分配上限，而不是 `ref[:cap]` 一刀切：一刀切会把靠后的
    文章整篇丢掉，答案引用它们时照样被判成编造 —— 那与「模型真读到了什么」不符。
    每块至少给 200 字，太短的块也要露个头。
    """
    pairs = [(b.get("article_id"), b.get("text") or "") for b in blocks if b]
    pairs = [(a, t) for a, t in pairs if t]
    if not pairs:
        return ""
    total = sum(len(t) for _, t in pairs)
    if total <= cap:
        return "\n\n".join(f"[{a}]\n{t}" for a, t in pairs)
    parts = []
    for aid, text in pairs:
        share = max(200, int(cap * len(text) / total))
        parts.append(f"[{aid}]\n{text[:share]}")
    return "\n\n".join(parts)


JUDGE_PROMPT = """你是严格的评分员。下面给你一个问题、一段知识库原文（参考答案）、以及系统给出的回答。

判断系统回答是否**忠实于原文**：
- 与原文矛盾 → 不忠实
- 出现了原文中没有的**具体**断言（数字、名称、步骤、结论）→ 不忠实（这是编造）
- 只是措辞不同、做了合理概括、或明确说明「笔记里没有」→ 忠实

问题：{question}

知识库原文：
{gold}

系统回答：
{answer}

只输出一个 JSON 对象，不要任何解释文字、不要代码块：
{{"faithful": true, "unsupported": [], "reason": "一句话"}}"""


def _judge(llm, question: str, gold: str, answer: str) -> dict:
    """LLM 裁判。**解析要稳、失败要计数** —— 上一版 74/76 次解析失败却没人发现，
    报告里那个「100% vs 0%」是纯噪声。"""
    try:
        reply = llm.chat(
            [
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=question, gold=gold, answer=answer[:JUDGE_ANSWER_CHARS]
                    ),
                }
            ]
        )
        raw = reply.content or ""
        if not raw.strip():
            # 空内容 = 推理把 token 预算用光了。没有 finish_reason 就只能靠猜，
            # 猜错过两轮，所以这里必须把线索带出来。
            return {
                "faithful": None,
                "error": f"内容为空（finish_reason={reply.finish_reason or '未知'}；"
                f"max_tokens={JUDGE_MAX_TOKENS}）",
            }
        # 模型可能包代码块或加前后缀：取第一个 JSON 对象
        raw = re.sub(r"```(?:json)?", "", raw)
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return {"faithful": None, "error": f"没找到 JSON：{raw[:80]!r}"}
        data = json.loads(m.group(0))
        return {
            "faithful": bool(data.get("faithful")),
            "unsupported": data.get("unsupported") or [],
            "reason": str(data.get("reason", ""))[:200],
        }
    except Exception as e:  # noqa: BLE001 - 裁判失败不该让整轮评测崩
        return {"faithful": None, "error": f"{type(e).__name__}: {e}"}


def _corpus_meta(skill_name: str):
    """只读语料拿 (titles, texts)，**不加载任何模型**（约 1 秒）。

    重算模式需要它：单轮臂按**标题**引用，要点术语也要从正文里提。
    """
    from rag_core.loader import DocumentLoader

    config = RAGConfig()
    docs = DocumentLoader(
        data_path=config.data_path,
        file_glob=config.file_glob,
        metadata_extractor=SKILLS[skill_name]().metadata_extractor,
    ).load()
    titles = {d.metadata.get("article_id"): d.metadata.get("title", "") for d in docs}
    texts = {d.metadata.get("article_id"): d.page_content for d in docs}
    return titles, texts


def _useful_term(term: str) -> bool:
    """能当要点的词：够长、不是纯数字/标点。"""
    if len(term) < 2:
        return False
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in term)
    alpha = sum(ch.isascii() and ch.isalpha() for ch in term)
    return has_cjk or alpha >= 3


# 推导要点时要先剥掉代码：否则 tf×idf 会把变量名（doc / scores / chunk / datetime）
# 当成「这篇文章的关键概念」，而正常回答里当然不会出现变量名 —— 指标被无谓压低。
_CODE_FENCE = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`[^`\n]*`")


def _strip_code(text: str) -> str:
    return _INLINE_CODE.sub(" ", _CODE_FENCE.sub(" ", text))


def _build_df(texts: dict) -> dict:
    from collections import defaultdict

    from rag_core.tokenizer import tokenize_for_bm25

    df = defaultdict(int)
    for text in texts.values():
        for tok in set(tokenize_for_bm25(_strip_code(text))):
            df[tok.lower()] += 1
    return df


def _derive_probes(ids, texts: dict, df: dict, top_n: int = 8) -> list:
    """从 gold 文章里自动挑「在这篇频繁、在别处罕见」的术语作为要点。

    **不由人挑** —— 手挑的要点会不自觉地往「答案里一定会出现」的方向选，
    等于自证。用 tf × idf 自动取前 N 个，选出来的词才是真正区分这篇文章的。
    派生时剥掉代码块，只取正文里的概念词。
    """
    import math
    from collections import Counter

    from rag_core.tokenizer import tokenize_for_bm25

    n_docs = max(len(texts), 1)
    tf: Counter = Counter()
    for aid in ids:
        tf.update(t for t in tokenize_for_bm25(_strip_code(texts.get(aid, ""))) if _useful_term(t))

    scored = []
    for term, count in tf.items():
        d = df.get(term.lower(), 1)
        if d > n_docs * 0.6:  # 太通用的词不区分文章
            continue
        scored.append((count * math.log(n_docs / max(d, 1)), term))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [t for _, t in scored[:top_n]]


def _score_row(row: dict, titles: dict) -> None:
    """按当前判分规则给一行的每个臂打分（纯函数，可反复重算）。"""
    gate = row["kind"] in GATE_KINDS
    manual = row["kind"] in MANUAL_KINDS
    expect = row.get("expect") or []
    for arm, a in row["arms"].items():
        got, why = _abstained(a.get("answer", ""), a.get("needs_human", False))
        a["abstained"], a["abstain_reason"] = got, why
        a["context_ok"] = bool(set(expect) & set(a.get("context_ids") or [])) if not gate else None
        form = ""
        for aid in expect:
            form = _cited(a.get("answer", ""), aid, titles.get(aid, "")) or form
        a["cited_form"] = form
        a["cited"] = bool(form)
        a["cited_right"] = bool(a["context_ok"] and a["cited"]) if not gate else None
        a["pure_abstain"] = bool(got and not a["cited"]) if not gate else None
        if manual:
            # partial 没有 gold，所以「给方向」要看它有没有指向语料里任意一篇
            a["cited_any"] = _cited_any(a.get("answer", ""), titles)
        if not gate and not manual:
            cov, missing = _coverage(a.get("answer", ""), row.get("probes") or [])
            a["coverage"], a["missing_terms"] = round(cov, 3), missing[:4]


def _summarize(detail, arms, judging: bool, seconds: dict) -> dict:
    stats = {
        a: {
            "n_ans": 0, "context_ok": 0, "cited": 0, "abstain": 0, "pure_abstain": 0,
            "cited_right": 0, "cite_form": {"id": 0, "title": 0}, "by_kind": {},
            "n_gate": 0, "declares_gap": 0, "silent_gap": 0,
            "n_manual": 0, "manual_cited": 0, "manual_gap": 0,
            "judged_ok": 0, "judge_fail": 0, "faithful": 0, "seconds": seconds.get(a, 0.0),
            "cov": [], "cov_shuffled": [],
        }
        for a in arms
    }
    for row in detail:
        gate = row["kind"] in GATE_KINDS
        manual = row["kind"] in MANUAL_KINDS
        for arm in arms:
            a = row["arms"].get(arm)
            if not a:
                continue
            s = stats[arm]
            if gate:
                s["n_gate"] += 1
                if a["abstained"]:
                    s["declares_gap"] += 1
                else:
                    s["silent_gap"] += 1
            elif manual:
                # 人工判定类：语料有相关内容但不完全回答。既不算可答也不算闸门。
                s["n_manual"] += 1
                s["manual_cited"] += int(bool(a.get("cited_any")))
                s["manual_gap"] += int(bool(a["abstained"]))
            else:
                s["n_ans"] += 1
                s["context_ok"] += int(bool(a["context_ok"]))
                s["cited"] += int(bool(a["cited"]))
                s["abstain"] += int(bool(a["abstained"]))
                s["pure_abstain"] += int(bool(a["pure_abstain"]))
                s["cited_right"] += int(bool(a["cited_right"]))
                if a["cited_form"]:
                    s["cite_form"][a["cited_form"]] += 1
                bk = s["by_kind"].setdefault(row["kind"], {"n": 0, "cited_right": 0, "context_ok": 0})
                bk["n"] += 1
                bk["cited_right"] += int(bool(a["cited_right"]))
                bk["context_ok"] += int(bool(a["context_ok"]))
            j = a.get("judge")
            if judging and j:
                if j.get("faithful") is None:
                    s["judge_fail"] += 1
                else:
                    s["judged_ok"] += 1
                    s["faithful"] += int(j["faithful"])
    # 要点覆盖率 + **打乱对照**：把答案配到下一题的要点上。
    # 对照不降下来就说明这个指标没有区分力，数字不能信。
    ans_rows = [r for r in detail if _is_answerable(r) and r.get("probes")]
    for i, row in enumerate(ans_rows):
        nxt = ans_rows[(i + 1) % len(ans_rows)]
        for arm in arms:
            a = row["arms"].get(arm)
            if not a:
                continue
            stats[arm]["cov"].append(a.get("coverage") or 0.0)
            stats[arm]["cov_shuffled"].append(
                _coverage(a.get("answer", ""), nxt.get("probes") or [])[0]
            )
    return stats


def _print_report(detail, arms, stats, judging: bool) -> None:
    print("\n【可答题】引对率（材料给了 + 答案引对了；**不等于答对了**）")
    head = f"  {'臂':<8}{'n':>4}{'上下文命中':>12}{'引对':>9}{'纯弃答':>9}{'引对率':>9}"
    if judging:
        head += f"{'裁判忠实':>10}{'裁判失败':>10}"
    print(head)
    for arm in arms:
        s = stats[arm]
        n = max(s["n_ans"], 1)
        line = (
            f"  {arm:<8}{s['n_ans']:>4}{s['context_ok'] / n:>11.1%}{s['cited'] / n:>9.1%}"
            f"{s['pure_abstain'] / n:>9.1%}{s['cited_right'] / n:>9.1%}"
        )
        if judging:
            line += f"{s['faithful'] / max(s['judged_ok'], 1):>10.1%}{s['judge_fail']:>10}"
        print(line)

    print("\n  按题型拆（引对率）：")
    for kind in ANSWERABLE:
        line = f"    {kind:<12}"
        for arm in arms:
            bk = stats[arm]["by_kind"].get(kind)
            line += f"{(bk['cited_right'] / bk['n'] if bk and bk['n'] else 0):>12.1%}"
        print(line)

    print("\n【要点覆盖率】答案用上了 gold 文章里多少「高区分度术语」（tf×idf 自动提取，非人挑）")
    print("  这是**覆盖率不是正确性** —— 只说明有没有用上关键概念，说明不了说得对不对")
    print(f"  {'臂':<8}{'要点覆盖':>10}{'打乱对照':>10}{'差值':>9}")
    for arm in arms:
        c, sh = stats[arm]["cov"], stats[arm]["cov_shuffled"]
        if not c:
            continue
        mc, ms = sum(c) / len(c), (sum(sh) / len(sh) if sh else 0.0)
        verdict = "有区分力" if mc - ms > 0.15 else "⚠ 区分力弱，别当结论"
        print(f"  {arm:<8}{mc:>9.1%}{ms:>10.1%}{mc - ms:>+9.1%}   {verdict}")
    print("  （打乱对照＝把答案配到下一题的要点上；对照不降就说明指标没区分力）")

    if judging:
        print("\n  关于「裁判忠实」怎么读（第 5 版，前 4 版都不可信）：")
        print("    · 已校准：真上下文 6/6 忠实 vs 打乱上下文 0/6，注入假细节探针 3/3 抓出；")
        print("      参照物 = **模型实际读到的正文（截断后）**，不是 gold。")
        print("    · 但它的**绝对准确率没有人工基准**（官方数据：LLM 判官与人工一致率约 75%~87%），")
        print("      所以请把这个数字读成「高/低」，不要读成「精确值」；失败条数在表里。")

    print("\n【闸门题】正确行为是声明「笔记里没有」")
    print(f"  {'臂':<8}{'n':>4}{'声明缺口':>10}{'未声明':>9}")
    for arm in arms:
        s = stats[arm]
        n = max(s["n_gate"], 1)
        print(f"  {arm:<8}{s['n_gate']:>4}{s['declares_gap'] / n:>9.1%}{s['silent_gap'] / n:>9.1%}")
    print("  「未声明」≠「编造」：诚实地说「没有专门章节，但散落的内容是这些」也算未声明。")
    print("  要判真编造看 `--judge` 的忠实度。")

    if any(stats[a]["n_manual"] for a in arms):
        print("\n【partial 类】语料有相关内容但不完全回答（既不算可答也不算闸门）")
        print("  理想行为：承认不完全 **且** 指出语料里最接近的方向")
        print(f"  {'臂':<8}{'n':>4}{'引到相关内容':>14}{'声明不完全':>12}")
        for arm in arms:
            s = stats[arm]
            n = max(s["n_manual"], 1)
            print(f"  {arm:<8}{s['n_manual']:>4}{s['manual_cited'] / n:>13.1%}{s['manual_gap'] / n:>11.1%}")

    print("\n【引用格式分布】两臂不一致本身就是问题（前端没法给标题加链接）")
    for arm in arms:
        f = stats[arm]["cite_form"]
        tot = max(f["id"] + f["title"], 1)
        print(f"  {arm:<8} article_id {f['id']} 次（{f['id'] / tot:.0%}） / 标题 {f['title']} 次（{f['title'] / tot:.0%}）")

    print("\n【成本】真实用量（来自 API 返回，非估算）")
    for arm in arms:
        c = sum(r["arms"][arm].get("calls", 0) for r in detail)
        pt = sum(r["arms"][arm].get("prompt_tokens", 0) for r in detail)
        ct = sum(r["arms"][arm].get("completion_tokens", 0) for r in detail)
        print(f"  {arm:<8} 调用 {c:>4} 次  prompt {pt:>9,} tok  completion {ct:>8,} tok  "
              f"耗时 {stats[arm]['seconds']:>6.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0 = 全量）")
    parser.add_argument(
        "--kind",
        default="",
        help="只跑这些类型，逗号分隔（如 partial / absent,mentioned）。空 = 全部",
    )
    parser.add_argument("--arms", default="single,agent")
    parser.add_argument("--judge", action="store_true", help="跑生成时顺带裁判")
    parser.add_argument("--rescore", action="store_true", help="用已保存答案重算确定性指标")
    parser.add_argument("--judge-only", action="store_true", help="只补裁判（用已保存答案）")
    parser.add_argument(
        "--shuffle-context",
        action="store_true",
        help="**对照臂**（配 --judge-only）：发答案本身，但参照上下文换成别题的。验证裁判有没有区分力",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="不跑评测，直接打印已保存报告里每题的答案与判分依据（人工复核用）",
    )
    parser.add_argument(
        "--report",
        default="",
        help="配合 --show 指定报告文件；--judge-only/--rescore 也认它 —— "
        "只补某个子集报告的裁判时用（不指定就动全量报告）",
    )
    args = parser.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    kinds = [k.strip() for k in args.kind.split(",") if k.strip()]

    if args.show:
        path = Path(args.report) if args.report else (
            SUBSET if SUBSET.exists() else REPORT
        )
        if not path.exists():
            raise SystemExit(f"找不到报告：{path}，先跑一次评测")
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"报告：{path}（{len(data['detail'])} 条，臂={data['arms']}）\n")
        for i, row in enumerate(data["detail"], 1):
            print(f"── {i}. [{row['kind']}] {row['question']}")
            for arm in data["arms"]:
                a = row["arms"].get(arm)
                if not a:
                    continue
                marks = []
                if a.get("cited") is not None:
                    marks.append(f"引对={a['cited']}")
                if a.get("coverage") is not None:
                    marks.append(f"要点覆盖={a['coverage']:.0%}")
                    if a.get("missing_terms"):
                        marks.append(f"缺={a['missing_terms']}")
                if a.get("abstained"):
                    marks.append(f"弃答({a.get('abstain_reason')})")
                print(f"  [{arm}] {' '.join(marks)}")
                print(f"    {a['answer'][:400]}")
            print()
        return

    if kinds and (args.rescore or args.judge_only):
        raise SystemExit(
            "--kind 只能用于「跑生成」。--rescore / --judge-only 是在整份报告上重算的，"
            "按类型过滤会把报告截断。想只重算某一类，改完判分规则后直接 --rescore 即可。"
        )
    if args.shuffle_context and not args.judge_only:
        raise SystemExit(
            "--shuffle-context 只配 --judge-only 用：对照臂复用**已经生成好的**答案，"
            "只多花裁判的钱；重新生成既慢又没意义（答案本身不用换）。"
        )

    if args.rescore or args.judge_only:
        # 默认动全量报告；--report 指到子集就只动子集（小样本验证裁判时用这个，
        # 否则一条 6 题的实验会把 76 条全量报告重新裁判一遍 = 白花钱）
        src = Path(args.report) if args.report else REPORT
        if not src.exists():
            raise SystemExit(f"没有可重算的报告：{src}，先跑一次生成")
        saved = json.loads(src.read_text(encoding="utf-8"))
        detail, arms = saved["detail"], saved["arms"]
        skill_name = os.getenv("EVAL_SKILL", "notes")
        # 答案是一次性生成的（贵），标注是会改的（便宜）——
        # 所以重算时把标注按当前 golden.jsonl 同步过来，不必重新生成。
        titles, texts = _corpus_meta(skill_name)
        synced = _sync_labels(detail, GOLDEN)
        df = _build_df(texts)
        for row in detail:
            row["probes"] = (
                _derive_probes(row.get("expect") or [], texts, df) if _is_answerable(row) else []
            )
        print(f"（标注已按 golden.jsonl 同步：{synced} 条的 kind/expect 有变化；"
              f"要点术语从语料自动提取）")
        seconds = {a: saved["stats"][a]["seconds"] for a in arms}
        judging = saved.get("judge", False)

        if args.judge_only:
            config = RAGConfig()
            config.validate()
            pipe = BasicRAGPipeline(config, SKILLS[os.getenv("EVAL_SKILL", "notes")]()).build()
            text_by_id = {d.metadata.get("article_id"): d.page_content for d in pipe.documents}
            judge_llm = OpenAICompatLLM(
                model=config.llm_model, base_url=config.llm_base_url,
                api_key=config.llm_api_key, temperature=0.0, max_tokens=JUDGE_MAX_TOKENS,
            )
            pairs = [
                (row, arm, row["arms"][arm])
                for row in detail
                if row["kind"] not in GATE_KINDS and row.get("expect")
                for arm in arms
                if row["arms"].get(arm)
            ]
            print(f"（只补裁判：{len(pairs)} 条答案）")
            print("  参照物 = **模型实际看到的上下文正文**（context_texts），不是 gold。")
            print("  早先用 gold 当参照，裁判把来自其他上下文文章的引用全判成编造，")
            print("  得出「忠实度 5.9%」这种假结论。")
            print(f"  参照物上限 {JUDGE_REF_CHARS} 字（按块比例分配，每篇都会露头）——"
                  "上限太小会把答案判成编造，这坑踩过。")
            refs, missing_ctx = [], 0
            for row, _arm, a in pairs:
                # 参照物优先用模型真读到的正文；旧报告没这个字段才退回文章全文
                blocks = a.get("context_texts") or []
                if not blocks:
                    missing_ctx += 1
                    blocks = [
                        {"article_id": x, "text": text_by_id.get(x, "")}
                        for x in (a.get("context_ids") or row["expect"])
                    ]
                refs.append(_fit_reference(blocks))
            sizes = [len(r) for r in refs if r]
            if sizes:
                print(f"  实际参照物 {min(sizes)}–{max(sizes)} 字")
            # **对照臂**：参照物轮转一位，每条拿到的是**别题**的上下文。
            if args.shuffle_context and len(refs) > 1:
                refs = _rotate(refs)
                print("  ⚠ **对照模式**：每条答案配的是别题的上下文，忠实度**应当明显下降**。")
                print("     不下降 = 裁判只会点头，它测的根本不是「忠实」。")
            for i, ((row, _arm, a), ref) in enumerate(zip(pairs, refs), 1):
                if ref:
                    a["judge"] = _judge(judge_llm, row["question"], ref, a["answer"])
                if i % 10 == 0:
                    print(f"  ...{i}/{len(pairs)}")
            if missing_ctx:
                print(
                    f"  ⚠ {missing_ctx} 条没有 context_texts（旧报告），退回用文章全文当参照 ——\n"
                    "     这比真实上下文**宽松**，重跑一次生成才有准确参照。"
                )
            judging = True
            saved["usage"]["judge"] = judge_llm.usage

        for row in detail:
            _score_row(row, titles)
        stats = _summarize(detail, arms, judging, seconds)
        print(f"（{'只补裁判' if args.judge_only else '重算'}模式：{len(detail)} 条，"
              f"{'确定性指标未调用 LLM' if not args.judge_only else '仅裁判调用了 LLM'}）")
        _print_report(detail, arms, stats, judging)
        # 对照臂绝不覆盖源报告：它的裁判结果是对着**错**上下文打的
        out = SHUFFLE if args.shuffle_context else src
        out.write_text(
            json.dumps({**saved, "judge": judging, "stats": stats, "detail": detail,
                        "titles": titles}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n已回写 {out}")
        if args.shuffle_context:
            print("（这是**对照臂**，真实报告没有被改动；看图："
                  f"python eval\\run_gen_eval.py --show --report {out.name}）")
        return

    config = RAGConfig()
    config.validate()
    pipe = BasicRAGPipeline(config, SKILLS[os.getenv("EVAL_SKILL", "notes")]()).build()
    title_by_id = {d.metadata.get("article_id"): d.metadata.get("title", "") for d in pipe.documents}
    text_by_id = {d.metadata.get("article_id"): d.page_content for d in pipe.documents}

    answer_llm = OpenAICompatLLM(
        model=config.llm_model, base_url=config.llm_base_url, api_key=config.llm_api_key,
        temperature=config.temperature, max_tokens=config.max_tokens,
    )
    judge_llm = OpenAICompatLLM(
        model=config.llm_model, base_url=config.llm_base_url, api_key=config.llm_api_key,
        temperature=0.0, max_tokens=JUDGE_MAX_TOKENS,
    )
    tools = build_tools(pipe, min_relevance=config.min_relevance)

    # 以下是「跑生成」路径（--rescore / --judge-only 已在上面 return）
    golden = [
        json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        golden = golden[: args.limit]
    if kinds:
        all_kinds = sorted({g["kind"] for g in golden})
        before = len(golden)
        golden = [g for g in golden if g["kind"] in kinds]
        if not golden:
            raise SystemExit(f"--kind {kinds} 没匹配到条目。现有类型：{all_kinds}")
        print(f"（--kind {'/'.join(kinds)}：{len(golden)}/{before} 条）")

    # **子集运行不能覆盖全量报告**：last_gen_report.json 是 --rescore 的依据，
    # 被 3 条 partial 覆盖掉就得重新花钱生成 78 条。所以子集另存一份。
    subset = bool(args.limit or kinds)
    out_path = SUBSET if subset else REPORT
    df = _build_df(text_by_id)
    detail, seconds = [], {a: 0.0 for a in arms}
    for i, item in enumerate(golden, 1):
        row = {
            "kind": item["kind"],
            "question": item["question"],
            "expect": item.get("expect") or [],
            "arms": {},
        }
        row["probes"] = (
            _derive_probes(row["expect"], text_by_id, df) if item["kind"] in ANSWERABLE else []
        )
        gold = "\n\n".join(text_by_id.get(a, "") for a in row["expect"])
        for arm in arms:
            t0 = time.time()
            if arm == "single":
                with get_openai_callback() as cb:
                    # 要**父文档正文**而不只是 id：判编造必须拿模型实际看到的上下文当参照
                    answer, parents = pipe.query_with_sources(row["question"])
                ctx = [d.metadata.get("article_id") for d in parents]
                ctexts = [
                    {
                        "tool": "single",
                        "article_id": d.metadata.get("article_id"),
                        "title": d.metadata.get("title"),
                        "text": d.page_content,
                    }
                    for d in parents
                ]
                calls, pt, ct = cb.successful_requests, cb.prompt_tokens, cb.completion_tokens
                needs_human, agent_cites = False, []
            else:
                before = dict(answer_llm.usage)
                res = RAGAgent(answer_llm, tools, max_steps=6).run(row["question"])
                answer = res.answer
                ctx = sorted({c.get("article_id") for c in res.citations if c.get("article_id")})
                # Agent 真正读到的片段/全文（citations 里没有正文）
                ctexts = res.contexts
                needs_human = res.needs_human
                agent_cites = [c.get("article_id") for c in res.citations]
                calls = answer_llm.usage["calls"] - before["calls"]
                pt = answer_llm.usage["prompt_tokens"] - before["prompt_tokens"]
                ct = answer_llm.usage["completion_tokens"] - before["completion_tokens"]
            dt = time.time() - t0
            seconds[arm] += dt
            row["arms"][arm] = {
                "answer": answer,
                "excerpt": answer[:160].replace("\n", " "),
                "context_ids": ctx,
                "context_texts": ctexts,
                "agent_citations": agent_cites,
                "needs_human": needs_human,
                "calls": calls,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "seconds": round(dt, 2),
            }
            if args.judge and row["kind"] not in GATE_KINDS and ctexts:
                # 参照物用**模型实际读到的上下文**，不是 gold —— 与 --judge-only 同口径。
                # 早先这里用 gold，凡是引用到上下文里其他文章的内容全被判成编造。
                row["arms"][arm]["judge"] = _judge(
                    judge_llm, row["question"], _fit_reference(ctexts), answer
                )
        _score_row(row, title_by_id)
        detail.append(row)
        if i % 5 == 0 or i == len(golden):
            print(f"  ...{i}/{len(golden)}")
    judging = args.judge

    stats = _summarize(detail, arms, judging, seconds)
    print(f"\nskill={os.getenv('EVAL_SKILL', 'notes')}  标注 {len(detail)} 条  "
          f"（可答 {sum(1 for g in detail if g['kind'] in ANSWERABLE)}，"
          f"闸门 {sum(1 for g in detail if g['kind'] in GATE_KINDS)}）  臂={arms}")
    _print_report(detail, arms, stats, judging)

    out_path.write_text(
        json.dumps(
            {
                "arms": arms,
                "judge": judging,
                "titles": title_by_id,
                "stats": stats,
                "usage": {"answer": answer_llm.usage, "judge": judge_llm.usage},
                "detail": detail,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n明细已写到 {out_path}（含每题答案与判分依据，便于审计）")
    if subset:
        print(f"⚠ 这是子集运行（{len(detail)} 条），所以**没有**动 {REPORT.name} ——")
        print("  全量报告是 --rescore 的依据，被覆盖就得重新花钱生成。")
        print("  看子集结果（含每题答案）：python eval\\run_gen_eval.py --show")


if __name__ == "__main__":
    main()
