"""统计 token 的文档频率（df），用来挑「只出现在一篇文章里」的罕见术语。

`rare_token` 类型的题目**必须由测量选出来**，不能凭感觉：上一版凭感觉选的
`MCP`、`mem0` 两个检索器都能命中，整组白测。只挑 df 极小的术语，BM25 才有
机会真正跑赢向量。

    python eval/token_rarity.py --max-df 2 --limit 40
    EVAL_SKILL=recipe python eval/token_rarity.py
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import bootstrap  # noqa: E402

bootstrap.prepare()

from dotenv import load_dotenv  # noqa: E402

load_dotenv(bootstrap.env_file_for(os.getenv("EVAL_SKILL", "notes")))

from rag_core import RAGConfig  # noqa: E402
from rag_core.loader import DocumentLoader  # noqa: E402
from rag_core.skills import SKILLS  # noqa: E402
from rag_core.tokenizer import tokenize_for_bm25  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-df", type=int, default=2, help="只列文档频率不超过它的词")
    parser.add_argument("--limit", type=int, default=40, help="打印前几个")
    parser.add_argument("--json", action="store_true", help="全量写 eval/rare_tokens.json")
    args = parser.parse_args()

    skill_name = os.getenv("EVAL_SKILL", "notes")
    config = RAGConfig()
    docs = DocumentLoader(
        data_path=config.data_path,
        file_glob=config.file_glob,
        metadata_extractor=SKILLS[skill_name]().metadata_extractor,
    ).load()

    # 一次遍历建 token -> 文章集合（别对每个词重新分词，99 篇 × N 词会跑到超时）
    # **必须转小写**，和 retriever._build_token_df 保持同一套 df 定义 ——
    # 否则这里量出 APPROVAL df=1、分类器算出 df=7，标注和分类器就会打架
    tok_to_docs: dict = defaultdict(set)
    for d in docs:
        aid = d.metadata.get("article_id") or d.metadata.get("source")
        for tok in set(tokenize_for_bm25(d.page_content)):
            tok_to_docs[tok.lower()].add(aid)

    rare = []
    for tok, where in tok_to_docs.items():
        if len(where) > args.max_df or len(tok) < 3:
            continue
        # 至少 3 个拉丁字母：排掉 00Z / 10s / 200K / 06d9 这类 ID、时间戳、容量
        if sum(ch.isascii() and ch.isalpha() for ch in tok) < 3:
            continue
        rare.append({"token": tok, "df": len(where), "in": sorted(where)})
    rare.sort(key=lambda x: (x["df"], x["token"]))

    print(f"skill={skill_name}  语料 {len(docs)} 篇  token {len(tok_to_docs)} 个")
    print(f"df<={args.max_df} 的候选 {len(rare)} 个，前 {args.limit} 个：\n")
    for r in rare[: args.limit]:
        print(f"  df={r['df']}  {r['token']:<34}{', '.join(r['in'])}")

    if args.json:
        out = Path(bootstrap.artifact_path("rare_tokens", skill_name, ".json", str(ROOT / "eval")))
        out.write_text(json.dumps(rare, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n全量已写到 {out}")


if __name__ == "__main__":
    main()
