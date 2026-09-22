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

用法（先有子集答案：`python eval/run_gen_eval.py --limit 6 --kind easy --arms single`）：
    python eval/judge_probe.py
    python eval/judge_probe.py --report eval/last_gen_subset.json --index 2

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

# 与原文**矛盾**的注入：题 1 的参照物里白纸黑字写着「如 512 token 一块」
INJECT_CONTRADICT = (
    "\n\n补充：本课程的知识库把 RAG 的分块大小固定为 777 token，"
    "检索阶段固定取 Top-7，这两项参数在所有章节中保持一致。"
)
# 原文没有、但听起来合理且不矛盾的注入 —— 这一类才是真编造的常见样子
INJECT_PLAUSIBLE = "\n\n补充：实践中最常用的向量库是 Milvus，生产环境通常部署 8 个检索节点。"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(g.SUBSET), help="子集报告（默认 last_gen_subset）")
    parser.add_argument("--arm", default="single")
    parser.add_argument("--index", type=int, default=0, help="用第几条做探针（0 = 第一条）")
    args = parser.parse_args()

    path = os.path.join(ROOT, args.report)
    if not os.path.exists(path):
        raise SystemExit(f"找不到报告 {path}\n先跑：python eval/run_gen_eval.py --limit 6 --kind easy")
    data = json.load(open(path, encoding="utf-8"))
    row = data["detail"][args.index]
    arm = row["arms"][args.arm]
    if not arm.get("context_texts"):
        raise SystemExit("这条答案没有 context_texts，探针需要「模型实际读到的正文」当参照")

    ref = g._fit_reference(arm["context_texts"])
    question = row["question"]
    print(f"探针题目：{question}")
    print(f"参照物 {len(ref)} 字；答案 {len(arm['answer'])} 字\n")

    cfg = RAGConfig()
    cfg.validate()
    llm = OpenAICompatLLM(
        model=cfg.llm_model, base_url=cfg.llm_base_url, api_key=cfg.llm_api_key,
        temperature=0.0, max_tokens=g.JUDGE_MAX_TOKENS,
    )

    probes = (
        ("① 原答案", arm["answer"], True),
        ("② 注入与原文矛盾的数字", arm["answer"] + INJECT_CONTRADICT, False),
        ("③ 注入原文没有但听起来合理", arm["answer"] + INJECT_PLAUSIBLE, False),
    )
    ok = True
    for name, answer, expect in probes:
        verdict = g._judge(llm, question, ref, answer)
        got = verdict.get("faithful")
        good = got is expect
        ok &= good
        print(f"{name}  期望 faithful={expect}  实际={got}  {'✓' if good else '✗ 裁判有问题'}")
        print(f"   依据：{str(verdict.get('error') or verdict.get('reason'))[:200]}\n")

    print("结论：" + ("裁判有区分力 ✓" if ok else "裁判校准不过关 ✗ —— 别信它给的忠实度"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
