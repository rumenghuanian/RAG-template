"""生成「闸门类标注人工复核表」：`eval/gate_review.md`。

为什么需要人工
--------------
`build_golden.py` 能机器校验**词汇层面**的属性（`probe.absent` 的词一篇都不能出现、
`probe.present` 的词至少出现一次），但**验不了「语料到底答不答得了」**。
所以闸门类标注是「机器验一半 + 人判断一半」。

这个脚本把人工判断所需的两类证据一次性摊开，避免你来回翻语料：

  A. **probe 词出现的原文**（mentioned 才有）—— 判断「是被解释了，还是只被点名」
  B. **实际检索命中的文章 + 片段 + 分数** —— 判断「光看这段能不能回答问题」

用法：
    python eval/build_review.py            # 写 eval/gate_review.md
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import bootstrap  # noqa: E402

bootstrap.prepare()

from dotenv import load_dotenv  # noqa: E402

load_dotenv(bootstrap.env_file_for(os.getenv("EVAL_SKILL", "notes")))

import json  # noqa: E402

from rag_core import BasicRAGPipeline, RAGConfig  # noqa: E402
from rag_core.loader import DocumentLoader  # noqa: E402
from rag_core.skills import SKILLS  # noqa: E402

WINDOW = 110          # 探针词前后各取多少字
SNIPPET = 260         # 检索片段截断长度
CATEGORIES = ("easy", "paraphrase", "rare_token", "multi")


def _windows(text: str, term: str, limit: int = 2):
    """把探针词每次出现的前后文都切出来，最多 limit 处。"""
    out, low, start = [], text.lower(), 0
    while len(out) < limit:
        i = low.find(term.lower(), start)
        if i < 0:
            break
        a, b = max(0, i - WINDOW), min(len(text), i + len(term) + WINDOW)
        out.append(("…" if a else "") + text[a:b].replace("\n", " ") + ("…" if b < len(text) else ""))
        start = i + len(term)
    return out


def main() -> None:
    skill_name = os.getenv("EVAL_SKILL", "notes")
    config = RAGConfig()
    config.validate()

    golden = [
        json.loads(line)
        for line in Path(bootstrap.artifact_path("golden", skill_name, ".jsonl", str(ROOT / "eval"))).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gate = [g for g in golden if g["kind"] in ("absent", "mentioned")]
    if not gate:
        raise SystemExit("golden.jsonl 里没有闸门类标注，先跑 eval/build_golden.py")

    docs = DocumentLoader(
        data_path=config.data_path,
        file_glob=config.file_glob,
        metadata_extractor=SKILLS[skill_name]().metadata_extractor,
    ).load()
    text_by_id = {d.metadata.get("article_id"): d.page_content for d in docs}

    pipe = BasicRAGPipeline(config, SKILLS[skill_name]()).build()

    lines = [
        "# 闸门类标注人工复核表",
        "",
        f"共 **{len(gate)}** 条（`absent` {sum(1 for g in gate if g['kind'] == 'absent')} 条、"
        f"`mentioned` {sum(1 for g in gate if g['kind'] == 'mentioned')} 条）。"
        f"语料 {len(docs)} 篇。",
        "",
        "## 为什么需要你判断",
        "",
        "`build_golden.py` 只校验**词汇层面**的属性：",
        "",
        "- `absent`：`probe.absent` 里的词**一篇都不能出现**（术语完全不存在）",
        "- `mentioned`：`probe.present` 至少出现一次，且 `probe.absent` 一篇都不出现"
        "（**被问的那一面**不存在）",
        "",
        "它**验不了「语料到底答不答得了」**——那需要读懂片段在讲什么。",
        "第一次做这件事时我手标了 8 条「语料外」，5 条其实是可答的，还基于错误标注得出过错误结论。",
        "",
        "## 判断标准",
        "",
        "每条问自己一句：**只看下面这些证据，能不能回答这个问题？**",
        "",
        "### `absent`（术语完全不出现）",
        "",
        "1. **这个词是否真的决定了可答性？** 如果问题里还有别的词能在语料里找到答案",
        "   （同义词、上位词、别称），那它其实**可答** → 改标可答并给出 `expect`。",
        "2. 别只看这一个词。例如「知识图谱和向量检索怎么配合」——`知识图谱` 不在语料里，",
        "   但如果语料讲了「图结构增强检索」，那它就该算可答。**看检索命中的片段再判。**",
        "",
        "### `mentioned`（术语出现，但语料没回答这个问题）",
        "",
        "看「probe 词出现的原文」，判断属于哪一类：",
        "",
        "| 类型 | 特征 | 例子 |",
        "|---|---|---|",
        "| 只被点名 | 出现在外链、对比表格、一句话带过里，**没有解释** | `Hub-and-Spoke` 只有一条维基链接 |",
        "| 具体的面没讲 | 实体讲了，但**被问的那个方面**没有（`probe.absent` 就是这个面） | 讲了 Redis 是组件，没讲 RDB/AOF |",
        "| 需要外部/实时数据 | 语料里有相关文本，但答案根本不在语料里 | 「今天天气」需要实时数据 |",
        "",
        "**最容易错的一类**：术语没被解释，但语料用**别的话**讲了同一件事。",
        "这时应判为**可答**，把命中的文章写进 `expect`。",
        "（判断依据就是下面「实际检索命中」那段——如果它读起来像在回答这个问题，就是可答。）",
        "",
        "## 怎么回填",
        "",
        "在每条的「判定」行上勾一个，或直接告诉我序号 + 结论。三种结果：",
        "",
        "- **保留**：语料确实答不了 → 保持 `absent` / `mentioned` 不变",
        "- **改可答**：→ 需要给出 `expect`（命中文章 id），我会把它移进可答题统计",
        "- **不确定**：单独标出来，先不计入统计",
        "",
        "---",
        "",
    ]

    for n, item in enumerate(gate, 1):
        probe = item.get("probe", {})
        tag = item["kind"] + (f"·{item['note']}" if item.get("note") else "")
        lines.append(f"### {n}. `{tag}` ｜ {item['question']}")
        lines.append("")
        bits = []
        for term in probe.get("absent", []):
            bits.append(f"`absent: {term}`（机器已验：0 篇）")
        for term in probe.get("present", []):
            hits = sorted(a for a, t in text_by_id.items() if term.lower() in t.lower())
            bits.append(f"`present: {term}`（机器已验：{len(hits)} 篇）")
        lines.append("- " + "　".join(bits))

        for term in probe.get("present", []):
            lines.append(f"- **语料里出现「{term}」的原文**：")
            shown = False
            for aid in sorted(a for a, t in text_by_id.items() if term.lower() in t.lower()):
                for snip in _windows(text_by_id[aid], term, limit=1):
                    lines.append(f"  - `{aid}`：{snip}")
                    shown = True
            if not shown:
                lines.append("  - （无）")

        hits = pipe.retrieve(item["question"], top_k=2)
        lines.append("- **实际检索命中（重排后）**：")
        if not hits:
            lines.append("  - （一篇都没命中）")
        for h in hits:
            cid = h.metadata.get("article_id")
            cos = h.metadata.get("vector_cos")
            rr = h.metadata.get("rerank_score")
            # 两个字段要**各自**格式化：BM25 单路命中的块本来就没有余弦，
            # 早先写成「两个都在才显示」，整行就只剩 id 了
            cos_s = f"{cos:.3f}" if cos is not None else "—(BM25 单路，无余弦)"
            rr_s = f"{rr:.3f}" if rr is not None else "—"
            lines.append(f"  - `{cid}`　余弦 {cos_s}　重排 {rr_s}")
            body = h.page_content[:SNIPPET].replace("\n", " ")
            lines.append(f"    > {body}…")
        lines.append("")
        lines.append("- **判定**：□ 保留　□ 其实可答（expect = ____________）　□ 不确定")
        lines.append("")
        lines.append("---")
        lines.append("")

    out = Path(bootstrap.artifact_path("gate_review", skill_name, ".md", str(ROOT / "eval")))
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"写了 {len(gate)} 条到 {out}")
    print("分类:", {k: sum(1 for g in gate if g["kind"] == k) for k in ("absent", "mentioned")})


if __name__ == "__main__":
    main()
