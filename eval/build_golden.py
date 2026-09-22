"""把 `eval/seeds.jsonl` 里的标注解析成真实 article_id，生成 `eval/golden.jsonl`。

标注只写 article_id / 标题的**唯一子串**（例如 `week5/33` 或 `重排序`），由脚本去语料里解析。
好处：
  - 换语料只改 `seeds.jsonl`，这个脚本和 `run_eval.py` 不用动（模板可复用）
  - 手写完整 id 很容易打错；匹配到 0 篇或多篇都会立刻报错，坏标注进不来

只读文档，不建索引，所以很快。
"""
import json
import os
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

from rag_core import RAGConfig  # noqa: E402
from rag_core.loader import DocumentLoader  # noqa: E402
from rag_core.skills import SKILLS  # noqa: E402
import scoring  # noqa: E402


# 类别口径只有一份，在 scoring 里（这里再抄一份就漏修过一次，见 scoring.bucket_of）
GATE_KINDS = tuple(scoring.GATE_KINDS)
# partial = 语料有相关内容但**不完全回答**该问题。它的定义属性（算不算「答了」）
# 本质上是人的判断，机器验不了，所以要求写 note 记录是谁、依据什么判的。
MANUAL_KINDS = tuple(scoring.MANUAL_KINDS)


def main() -> None:
    skill_name = os.getenv("EVAL_SKILL", "notes")
    config = RAGConfig()
    loader = DocumentLoader(
        data_path=config.data_path,
        file_glob=config.file_glob,
        metadata_extractor=SKILLS[skill_name]().metadata_extractor,
    )
    docs = loader.load()
    catalog = [
        {"id": d.metadata.get("article_id") or d.metadata.get("source"), "title": d.metadata.get("title", "")}
        for d in docs
    ]
    # probe 校验需要「哪些文章含这个字符串」，统一小写做子串匹配
    article_text = {
        c["id"]: d.page_content.lower() for c, d in zip(catalog, docs)
    }

    def resolve(spec: str) -> str:
        hits = [c for c in catalog if spec in (c["id"] or "") or spec in c["title"]]
        if len(hits) != 1:
            raise SystemExit(
                f"标注 {spec!r} 匹配到 {len(hits)} 篇，必须唯一匹配。"
                f"命中示例：{[h['id'] for h in hits[:3]]}"
            )
        return hits[0]["id"]

    def containing(term: str):
        t = term.lower()
        return sorted(aid for aid, text in article_text.items() if t in text)

    def check_probe(kind: str, question: str, probe: dict) -> str:
        """把「术语在不在语料里」变成机器校验 —— 这是闸门类标注的**定义属性**。

        第一次做闸门评测时我手标了 8 条「语料外」，后来发现 5 条其实可答
        （`Kubernetes 部署 Redis` 余弦 0.7808 不是闸门失灵，是那篇文章真在讲这件事），
        基于错误标注还得出过错误结论。所以现在标注必须自带可验证的证据：

          absent     → probe["absent"] 里的词**一篇都不能出现**（术语完全不存在）
          mentioned  → probe["present"] 至少出现一次（实体被提到）
                       且 probe["absent"] 里的词一篇都不出现
                       （**被问的那一面**不存在 —— 这才让「答不了」不再只靠人眼判断）

        注意它校验的是「词汇层面」的属性，不是「语料到底答不答得了」：
        像「今天天气怎么样」是数据时效问题（需要实时数据），词汇上没法表达。
        """
        bad_present = [t for t in probe.get("present", []) if not containing(t)]
        bad_absent = [t for t in probe.get("absent", []) if containing(t)]
        problems = []
        if bad_present:
            problems.append(f"标为「语料里有」但实际不存在：{bad_present}")
        if bad_absent:
            problems.append(
                "标为「语料里没有」但实际存在："
                + ", ".join(f"{t}（{len(containing(t))} 篇：{containing(t)[:2]}）" for t in bad_absent)
            )
        if problems:
            raise SystemExit(f"[{kind}] {question}\n  " + "\n  ".join(problems))
        absent_n = len(probe.get("absent", []))
        present_n = len(probe.get("present", []))
        return f"absent={absent_n} present={present_n}"

    seeds_path = Path(
        bootstrap.artifact_path("seeds", skill_name, ".jsonl", str(ROOT / "eval"))
    )
    if not seeds_path.exists():
        raise SystemExit(
            f"找不到标注文件：{seeds_path}\n"
            f"  每个 skill 有自己的一套标注（否则换语料会把上一套覆盖掉）。\n"
            f"  要跑 skill={skill_name}，先建 {seeds_path.name}：\n"
            f'    每行一个 JSON：{{"kind":"easy","question":"...","expect":["<article_id 的唯一子串>"]}}\n'
            f"  可用的 kind：easy / paraphrase / rare_token / multi"
            f"（可答）；absent / mentioned（闸门，要带 probe）；partial（人工判定，要带 note）"
        )
    seeds = [
        json.loads(line)
        for line in seeds_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    rows = []
    for s in seeds:
        kind, question = s["kind"], s["question"]
        probe = s.get("probe") or {}
        if kind in GATE_KINDS:
            if not probe:
                raise SystemExit(
                    f"[{kind}] {question}\n  闸门类标注必须带 probe，否则无法校验它是哪一类。"
                )
            if kind == "absent" and not probe.get("absent"):
                raise SystemExit(f"[absent] {question}\n  至少要有一个 probe['absent'] 词。")
            if kind == "mentioned" and not probe.get("present"):
                raise SystemExit(f"[mentioned] {question}\n  至少要有一个 probe['present'] 词。")
            check_probe(kind, question, probe)
        elif kind in MANUAL_KINDS:
            if probe:
                raise SystemExit(f"[{kind}] {question}\n  人工判定类不写 probe（它验不了）。")
            if not s.get("note"):
                raise SystemExit(
                    f"[{kind}] {question}\n  必须写 note 记录人工判定依据 —— 否则和「没标注过」没区别。"
                )
        elif probe:
            raise SystemExit(f"[{kind}] {question}\n  只有闸门类（absent/mentioned）才写 probe。")

        row = {
            "kind": kind,
            "question": question,
            "expect": [resolve(spec) for spec in s["expect"]],
        }
        if probe:
            row["probe"] = probe
        # note 只作文档用（例如 absent 的 far / near），不参与校验
        if s.get("note"):
            row["note"] = s["note"]
        rows.append(row)

    out = Path(bootstrap.artifact_path("golden", skill_name, ".jsonl", str(ROOT / "eval")))
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )

    counts: dict = {}
    for r in rows:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    print(f"skill={skill_name}  语料 {len(docs)} 篇")
    print(f"写了 {len(rows)} 条到 {out}（闸门类 probe 全部通过校验）")
    print("分类:", counts)


if __name__ == "__main__":
    main()
