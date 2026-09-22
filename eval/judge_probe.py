"""裁判（`run_gen_eval.py --judge`）的**校准探针**：改裁判 prompt / 参照物口径后跑一次。

为什么需要它
------------
LLM 裁判本身是**被测对象**。它会以两种方式坏掉，而且两种都会给出漂亮的假结论：

  ① 恒点头：对什么答案都说「忠实」→ 忠实度 100%，看着像好消息。
  ② 恒摇头：对什么答案都说「不忠实」→ 忠实度 0%，看着像系统全在编造。
     （实测过一次：参照物被截到 3000 字，真上下文和打乱上下文**都是 0/6**。）

所以裁判必须有**两个方向的对照**：
  - `--shuffle-context`（在 run_gen_eval 里）：配错上下文，忠实度应当掉下来 → 抓 ①
  - 本探针：把**已知忠实的答案人为弄坏**，看它抓不抓得住 → 抓 ②，并且测灵敏度

判据（三行都符合才算裁判可用）：
  ① 原答案        → `True`
  ② 注入与原文矛盾的具体数字 → `False`
  ③ 注入原文没有但听起来合理的细节 → `False`

⚠️ **探针自己也会坏，而且坏起来同样像「裁判坏了」**（真踩过一次）
----------------------------------------------------------------
种子必须是一条**已知忠实**的答案，而「已知」只能来自报告里存下来的裁判结论。
若那份报告是**修「记下未截断全文」那个 bug 之前**生成的，它记的上下文比模型实收的多：
裁判拿着多出来的那段一对账，就把「模型说这里没有」判成与原文矛盾 → ① 判出 `False`，
看着像裁判坏了，其实是**种子数据坏了**（旧报告 24,567 字、旧裁判结论还是忠实；
同一份答案现在的裁判判不忠实，因为它读到了模型根本没看到的「三、引用归属实现」）。

所以有两道闸，都在跑之前拦：
  1. 默认报告指向**修好之后**生成的全量报告（`REPORT`）；`--index` 不指定时
     **自动挑一条存有 `faithful=True` 的答案**当种子；指定了也要求它确实是忠实的。
  2. 按 token 预算核一遍：`context_texts` 的总 token 超过 `context_max_tokens`
     就说明记的不是实收正文，**直接拒绝跑**（实测旧报告 3/6 超预算，新报告 0/78）。

用法（报告里必须已经有裁判结论：`python eval/run_gen_eval.py --judge`）：
    python eval/judge_probe.py
    python eval/judge_probe.py --report eval/last_gen_report.json --index 2

退出码：0 = 裁判有区分力；1 = 抓不住注入；2 = 种子/报告有问题（**先别改裁判**）；
3 = 裁判没给出结论（量具没读数，调大 `JUDGE_MAX_TOKENS` 再跑）。

⚠️ 会调 3 次 LLM（探针本身要花钱，但比全量重跑便宜几个数量级）。
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

import bootstrap  # noqa: E402

bootstrap.prepare()
from dotenv import load_dotenv  # noqa: E402

load_dotenv(bootstrap.env_file_for(os.getenv("EVAL_SKILL", "notes")))
os.chdir(ROOT)

import run_gen_eval as g  # noqa: E402
from agent import OpenAICompatLLM  # noqa: E402
from rag_core import RAGConfig  # noqa: E402
from rag_core.tokenizer import estimate_tokens  # noqa: E402

# 与原文**矛盾**的注入：参照物里白纸黑字写着「如 512 token 一块」
INJECT_CONTRADICT = (
    "\n\n补充：本课程的知识库把 RAG 的分块大小固定为 777 token，"
    "检索阶段固定取 Top-7，这两项参数在所有章节中保持一致。"
)
# 原文没有、但听起来合理且不矛盾的注入 —— 这一类才是真编造的常见样子
INJECT_PLAUSIBLE = "\n\n补充：实践中最常用的向量库是 Milvus，生产环境通常部署 8 个检索节点。"


def _refuse(msg: str) -> None:
    """种子/报告有问题，**先别改裁判** —— 和「裁判抓不住」（退出码 1）分开报。

    注意别用 `raise SystemExit("说明")`：字符串参数会让退出码变成 1，
    和「裁判抓不住」撞在一起，CI 里分不出是哪种坏。
    """
    print(msg)
    sys.exit(2)


def seed_candidates(detail, arm="single"):
    """哪些条目能当种子：有答案、有「模型实际读到的正文」、且存下来的裁判结论是忠实。"""
    out = []
    for i, row in enumerate(detail):
        a = (row.get("arms") or {}).get(arm) or {}
        if not a.get("context_texts") or not a.get("answer"):
            continue
        if (a.get("judge") or {}).get("faithful") is True:
            out.append(i)
    return out


def pick_seed(detail, arm="single", index=None):
    """选探针种子。不指定 `index` 就自动挑一条已知忠实的。

    指定了也要求它**确实**是忠实的：拿一条自己就不忠实的答案去当「已知忠实」的
    基准，① 必然判 `False`，结论就变成「裁判坏了」—— 那是假警报。
    """
    if index is not None:
        row = detail[index]
        a = (row.get("arms") or {}).get(arm) or {}
        if not a.get("context_texts"):
            _refuse(f"第 {index} 条没有 context_texts，探针需要「模型实际读到的正文」当参照")
        verdict = (a.get("judge") or {}).get("faithful")
        if verdict is not True:
            _refuse(
                f"第 {index} 条存下来的裁判结论是 {verdict!r}，不是忠实。\n"
                "拿它当「已知忠实」的种子，① 必然判 False，那会变成假警报。\n"
                "换一条 --index，或者不指定 --index 让脚本自动挑。"
            )
        return index
    cands = seed_candidates(detail, arm)
    if not cands:
        _refuse(
            "这份报告里没有「存有裁判结论、且判为忠实」的条目，没法当基准。\n"
            "先跑：python eval/run_gen_eval.py --judge"
        )
    return cands[0]


def context_over_budget(context_texts, budget):
    """记下来的上下文，是不是**模型实收的那一份**？

    实收正文必然 ≤ `context_max_tokens`（拼接时按预算截断并追加「…（内容过长，已截断）」）。
    超过预算就说明记的是**未截断的全文** —— 那是修 bug 之前的报告，拿它当参照会误判。
    返回 `(是否超预算, 实际 token 数)`。
    """
    tokens = sum(estimate_tokens(str(t.get("text", ""))) for t in (context_texts or []))
    return tokens > budget, tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(g.REPORT), help="报告（默认全量报告）")
    parser.add_argument("--arm", default="single")
    parser.add_argument("--index", type=int, default=None,
                        help="用第几条做探针（不指定 = 自动挑一条已知忠实的）")
    args = parser.parse_args()

    path = os.path.join(ROOT, args.report)
    if not os.path.exists(path):
        _refuse(f"找不到报告 {path}\n先跑：python eval/run_gen_eval.py --judge")
    data = json.load(open(path, encoding="utf-8"))
    if args.arm not in (data.get("arms") or []):
        _refuse(f"报告里没有 {args.arm} 臂（有：{data.get('arms')}）")

    cfg = RAGConfig()
    cfg.validate()

    idx = pick_seed(data["detail"], args.arm, args.index)
    row = data["detail"][idx]
    arm = row["arms"][args.arm]
    over, tokens = context_over_budget(arm["context_texts"], cfg.context_max_tokens)

    print(f"种子：{os.path.relpath(path, ROOT)} 第 {idx} 条（{row['kind']}）—— {row['question']}")
    print(f"      存下来的裁判结论 = 忠实（「已知忠实」的来源）；答案 {len(arm['answer'])} 字")
    print(f"      记下的上下文 {tokens} token / 预算 {cfg.context_max_tokens}"
          f"（{len(arm['context_texts'])} 篇）")
    if over:
        _refuse(
            "拒绝跑：记下的上下文**超过**上下文预算，说明它不是模型实收的那一份。\n"
            "这种报告多半是修「记下未截断全文」之前生成的，拿它当参照会把「模型说这里没有」\n"
            "判成矛盾，看着像裁判坏了。\n"
            "→ 换 --report（默认已指向全量报告），或先重跑生成。"
        )

    ref = g._fit_reference(arm["context_texts"])
    question = row["question"]
    print(f"      参照物 {len(ref)} 字\n")

    llm = OpenAICompatLLM(
        model=cfg.llm_model, base_url=cfg.llm_base_url, api_key=cfg.llm_api_key,
        temperature=0.0, max_tokens=g.JUDGE_MAX_TOKENS,
    )

    probes = (
        ("① 原答案", arm["answer"], True),
        ("② 注入与原文矛盾的数字", arm["answer"] + INJECT_CONTRADICT, False),
        ("③ 注入原文没有但听起来合理", arm["answer"] + INJECT_PLAUSIBLE, False),
    )
    results = []
    for name, answer, expect in probes:
        verdict = g._judge(llm, question, ref, answer)
        got = verdict.get("faithful")
        mark = "✓" if got is expect else ("？没读数" if got is None else "✗")
        print(f"{name}  期望 faithful={expect}  实际={got}  {mark}")
        print(f"   依据：{str(verdict.get('error') or verdict.get('reason'))[:200]}\n")
        results.append((name, got, expect, verdict))

    first = results[0]
    if first[1] is None:
        print("结论：裁判**没给出结论**（不是判错）—— " + str(first[3].get("error")))
        print("     多半是推理把 token 预算吃光。调大 JUDGE_MAX_TOKENS 或调小参照物再跑。")
        sys.exit(3)
    if first[1] is not True:
        print("结论：① 原答案被判「不忠实」→ **先别改裁判**，按顺序查三件事：")
        print("      a) 这条种子答案本身在当前口径下就是不忠实的（比如它错说「笔记里没有」，而原文有）；")
        print("      b) 报告是修「记未截断全文」之前生成的（本脚本已按 token 预算拦一次，仍可疑就重跑生成）；")
        print("      c) 裁判真的变严了：拿这条答案人工看一遍，再决定改裁判还是换种子。")
        sys.exit(2)

    bad = [r for r in results[1:] if r[1] is not False]
    if bad:
        for name, got, _expect, verdict in bad:
            if got is None:
                print(f"{name}：裁判没给出结论（{verdict.get('error')}）"
                      "→ 量具没读数，不是「抓不住」；调大 JUDGE_MAX_TOKENS 再跑")
            else:
                print(f"{name}：裁判判成**忠实** → 这条路是抓不住编造")
        print("结论：裁判校准不过关 ✗ —— 别信它给的忠实度")
        sys.exit(1)

    print("结论：裁判有区分力 ✓（原答案忠实；两种注入的假细节都被抓出）")
    sys.exit(0)


if __name__ == "__main__":
    main()
