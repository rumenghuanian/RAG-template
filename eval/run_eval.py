"""检索层评测：向量 / BM25 / RRF 各自的 hit@5，**按题目类型拆开看**。

只看总分会被平均数骗过去（一份照标题改写的题集会让所有模式都是 100%）。
所以要按类型拆：
  easy        照标题改写 —— 基线，应该都很高
  paraphrase  换种说法、避开标题词 —— 考察语义召回，向量该赢
  rare_token  只出现在少数文章里的罕见术语（df 取自已索引语料，实测 df ≤ 3）
              —— 考察字面召回，BM25 该赢
  multi       答案散在 2~3 篇 —— 考察覆盖，看 cover@5 而不是 hit@5

另有两类**不参与 hit@5**，专门用来验「闸门能否判断语料里没有这件事」：
  absent      术语在语料里完全不出现 —— 理应被拦下
  mentioned   术语出现了，但语料并没有回答这个问题。例：「天气」是工具调用示例的标配
              例子（28 篇提到），但「今天天气怎么样」需要实时数据 —— **任何基于文本相似度
              的判据都必然失败**（实测余弦 0.42、重排 0.96，都拦不住）

**不调用 LLM**，所以可以随便跑、随便调参：
    python eval/run_eval.py --json
    EVAL_SKILL=recipe python eval/run_eval.py
"""
import argparse
import json
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

from rag_core import BasicRAGPipeline, RAGConfig  # noqa: E402
from rag_core.skills import SKILLS  # noqa: E402
import scoring  # noqa: E402

TOP_N = 5
K_SWEEP = [1, 10, 60, 100]
# 类别口径只有一份，在 scoring 里 —— 各脚本各抄一份的结果就是漏修（见 scoring.bucket_of）
ANSWERABLE = list(scoring.ANSWERABLE_KINDS)
# 不参与 hit@5 的两类，只用来验闸门
GATE_KINDS = list(scoring.GATE_KINDS)
# 人工判定类：语料有相关内容但不完全回答（定义属性验不了，靠 note 记录依据）
MANUAL_KINDS = list(scoring.MANUAL_KINDS)


def top_articles(docs, n=TOP_N):
    """按文章去重取前 n 篇 —— 与 agent 实际看到的一致（retrieve() 也是这么做的）。

    **没有 id 就报错，不许静默塌成 `[None]`**：换领域时踩过 —— skill 不产出 article_id，
    这里会把 5 篇去重成 `[None]`，于是 hit@5 全线 0%（连"宫保鸡丁怎么做"都是 0），
    而报告看起来只是"效果很差"，根本看不出是口径坏了。加载期已有契约校验，
    这里是第二道防线。
    """
    seen, out = set(), []
    for d in docs:
        aid = d.metadata.get("article_id")
        if not aid:
            raise RuntimeError(
                f"检索结果缺少 article_id（source={d.metadata.get('source')!r}）。\n"
                "  skill 的 metadata_extractor 必须产出规范字段 article_id/title，"
                "否则评测与引用全部失效。"
            )
        if aid in seen:
            continue
        seen.add(aid)
        out.append(aid)
        if len(out) >= n:
            break
    return out


def gate_subgroups(gate_items, kinds=GATE_KINDS):
    """闸门类按 `note` 再分组（absent 分 far/near）。纯函数，便于单测。

    **空组要跳过**：某个闸门类一条标注都没有时，这里会生成一个成员为空的组，
    下游 `sum(...)/len(vals)` 直接 ZeroDivisionError，整个评测崩掉。
    换任何新领域，第一份标注集都不太可能四类齐全 —— 实测就崩在这。
    """
    groups = []
    for kind in kinds:
        notes = sorted({g.get("note", "") for g in gate_items if g["kind"] == kind})
        for note in notes:
            members = [
                g for g in gate_items if g["kind"] == kind and g.get("note", "") == note
            ]
            if not members:
                continue
            groups.append((f"{kind}·{note}" if note else kind, members))
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="写明细到 eval/last_report.json")
    args = parser.parse_args()

    skill_name = os.getenv("EVAL_SKILL", "notes")
    golden = [
        json.loads(line)
        for line in Path(bootstrap.artifact_path("golden", skill_name, ".jsonl", str(ROOT / "eval"))).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    config = RAGConfig()
    config.validate()
    pipe = BasicRAGPipeline(config, SKILLS[skill_name]()).build()
    rt = pipe.retriever

    modes = ["vector", "bm25", "adaptive", "rerank"] + [f"rrf_k{k}" for k in K_SWEEP]
    per_kind = {
        k: {"n": 0, "hit": {m: 0 for m in modes}, "cover": {m: 0.0 for m in modes}}
        for k in ANSWERABLE
    }
    plan_stats: dict = {}
    gate_items, manual_items, overlaps, detail = [], [], [], []

    for item in golden:
        kind, q = item["kind"], item["question"]
        v = rt._vector_search(q, None)
        b = rt._bm25_search(q, None)
        overlaps.append(
            len({d.metadata.get("chunk_id") for d in v} & {d.metadata.get("chunk_id") for d in b})
            / max(len(v), 1)
        )
        best_cos = max((d.metadata.get("vector_cos") or 0.0) for d in v) if v else 0.0

        got = {"vector": top_articles(v), "bm25": top_articles(b)}
        # adaptive = 按 plan() 动态选权重/候选池；取全量融合结果再按文章去重到 TOP_N，
        # 与其它模式一样在「输出 5 篇」这个口径上比
        plan = rt.plan(q)
        # 两臂必须看**同一份候选集**，否则比的是池深而不是排序。
        # 池深现已与 top_k 解耦（由 rerank_candidates 决定，0 = 不截断），
        # 所以这里统一取全量：生产 retrieve(top_k=5) 也是这样重排整个 RRF 池的。
        got["adaptive"] = top_articles(rt.hybrid_search(q, top_k=10**6, rerank=False))
        # rerank = 同一份候选池，交给 cross-encoder 重新排序
        ranked = rt.hybrid_search(q, top_k=10**6, rerank=True)
        got["rerank"] = top_articles(ranked)
        # 闸门口径必须和**生产一致**：`search_notes` 取的是 `retrieve()` 返回的那 5 篇里的
        # 最高向量余弦。注意这 5 篇可能**全是 BM25 单路命中**（没有 vector_cos 字段），
        # 那样闸门读到的就是 0.0 —— 用 `_vector_search` 的 top-1 余弦衡量会比生产乐观。
        gate_cos, gate_rr, seen_a = 0.0, 0.0, set()
        for d in ranked:
            aid = d.metadata.get("article_id")
            if aid in seen_a:
                continue
            seen_a.add(aid)
            gate_cos = max(gate_cos, d.metadata.get("vector_cos") or 0.0)
            gate_rr = max(gate_rr, d.metadata.get("rerank_score") or 0.0)
            if len(seen_a) >= TOP_N:
                break
        for k in K_SWEEP:
            got[f"rrf_k{k}"] = top_articles(rt._rrf_rerank(list(v), list(b), k=k))

        plan_stats.setdefault(kind, {}).setdefault(plan["mode"], 0)
        plan_stats[kind][plan["mode"]] += 1

        if kind in GATE_KINDS:
            gate_items.append(
                {
                    "kind": kind,
                    "note": item.get("note", ""),
                    "q": q,
                    "gate_cos": round(gate_cos, 4),
                    "gate_rr": round(gate_rr, 4),
                }
            )
        elif kind in MANUAL_KINDS:
            # 人工判定类既不算可答（没有干净的 gold，折进去会变成循环标注），
            # 也不算闸门类（它不是「语料里没有」）。单独统计。
            manual_items.append(
                {
                    "kind": kind,
                    "q": q,
                    "note": item.get("note", ""),
                    "gate_cos": round(gate_cos, 4),
                    "gate_rr": round(gate_rr, 4),
                }
            )
        else:
            bucket = per_kind.setdefault(
                kind,
                {"n": 0, "hit": {m: 0 for m in modes}, "cover": {m: 0.0 for m in modes}},
            )
            bucket["n"] += 1
            for m in modes:
                if set(item["expect"]) & set(got[m]):
                    bucket["hit"][m] += 1
                if item["expect"]:
                    bucket["cover"][m] += len(set(item["expect"]) & set(got[m])) / len(
                        item["expect"]
                    )

        detail.append(
            {
                "kind": kind,
                "question": q,
                "expect": item["expect"],
                # max_cos = 向量 top-1（诊断用）；gate_cos/gate_rr = 生产闸门实际读到的值
                "max_cos": round(best_cos, 4),
                "gate_cos": round(gate_cos, 4),
                "gate_rr": round(gate_rr, 4),
                "got": got,
            }
        )

    total = sum(v["n"] for k, v in per_kind.items() if k in ANSWERABLE)
    gate_desc = "、".join(
        f"{k} {sum(1 for g in gate_items if g['kind'] == k)} 条" for k in GATE_KINDS
    )
    print(f"\nskill={skill_name}  标注 {len(golden)} 条，其中可答 {total} 条；闸门类：{gate_desc}")
    print(f"两路平均重叠率 {sum(overlaps) / len(overlaps):.1%}\n")

    head = f"{'类型':<12}{'n':>3}"
    for m in ["vector", "bm25", "rrf_k60", "adaptive", "rerank"]:
        head += f"{m:>10}"
    print(head)
    print("-" * len(head))
    for kind in ANSWERABLE:
        b_ = per_kind.get(kind)
        if not b_ or not b_["n"]:
            continue
        row = f"{kind:<12}{b_['n']:>3}"
        for m in ["vector", "bm25", "rrf_k60", "adaptive", "rerank"]:
            row += f"{b_['hit'][m] / b_['n']:>9.1%} "
        print(row)

    print("\n自适应（plan）相对固定权重 RRF 的增益：")
    for kind in ANSWERABLE:
        b_ = per_kind.get(kind)
        if not b_ or not b_["n"]:
            continue
        delta = (b_["hit"]["adaptive"] - b_["hit"]["rrf_k60"]) / b_["n"]
        print(f"  {kind:<12}{delta:>+7.1%}")

    print("\n重排序相对自适应融合的增益（同一份候选池，只换排序）：")
    for kind in ANSWERABLE + ["总体"]:
        if kind == "总体":
            before = sum(per_kind[k]["hit"]["adaptive"] for k in ANSWERABLE)
            after = sum(per_kind[k]["hit"]["rerank"] for k in ANSWERABLE)
        else:
            b_ = per_kind.get(kind)
            if not b_ or not b_["n"]:
                continue
            before, after = b_["hit"]["adaptive"], b_["hit"]["rerank"]
        base = before / total if kind == "总体" else before / b_["n"]
        new = after / total if kind == "总体" else after / b_["n"]
        print(f"  {kind:<12}{new - base:>+7.1%}   ({base:.1%} → {new:.1%})")

    print("\n策略分类校验（plan 判定的 mode vs 标注类型）：")
    for kind in ANSWERABLE:
        stats = plan_stats.get(kind, {})
        got = ", ".join(f"{m} {c}/{sum(stats.values())}" for m, c in sorted(stats.items()))
        print(f"  {kind:<12}{got}")

    print("\nk 敏感性（全部可答题目）：")
    for k in K_SWEEP:
        got = sum(per_kind[c]["hit"][f"rrf_k{k}"] for c in ANSWERABLE)
        print(f"  k={k:<4} hit@5 {got / total:>6.1%}")

    multi = per_kind.get("multi")
    if multi and multi["n"]:
        print(f"\nmulti 覆盖率@5（期望的 2~3 篇里有多少进了前 5，n={multi['n']}）：")
        for m in ["vector", "bm25", "rrf_k60", "adaptive", "rerank"]:
            print(f"  {m:<10}{multi['cover'][m] / multi['n']:>7.1%}")
        print("  （上面主表的 multi hit@5 是「至少命中 1 篇」，偏宽松，决策请看覆盖率）")

    # ---------- 闸门可行性：阈值扫描 + AUC ----------
    # 目标：把「重排分数/余弦看着能分开」变成「阈值 t 时拦下多少、误伤多少」。
    # 分数 **越低越像闸门类**，所以判据方向是 score < t 判为「语料里没有」。
    def _auc(gate_vals, ans_vals):
        """P(闸门项分数 < 可答项分数)。1.0 = 完全可分，0.5 = 等于瞎猜，<0.5 = 方向反了。"""
        if not gate_vals or not ans_vals:
            return None
        wins = sum(
            1.0 if g < a else (0.5 if g == a else 0.0)
            for g in gate_vals
            for a in ans_vals
        )
        return wins / (len(gate_vals) * len(ans_vals))

    if gate_items:
        ans = [
            d
            for d in detail
            if d["kind"] not in GATE_KINDS and d["kind"] not in MANUAL_KINDS
        ]
        # 闸门类分组：absent 还要分 far（完全域外）和 near（领域内但语料没有）
        subgroups = gate_subgroups(gate_items)

        print(f"\n闸门可行性：阈值扫描（口径＝生产实际读到的值：该问题前 5 篇里的最高分；"
              f"可答 {len(ans)} 条）")

        for field, judge in (
            ("gate_cos", "向量余弦"),
            ("gate_rr", "重排分数"),
        ):
            ans_vals = [d[field] for d in ans]
            if not ans_vals:
                print(f"\n  判据 = {judge}（没有可答题，跳过阈值扫描）")
                continue
            print(f"\n  判据 = {judge}")
            header = f"    {'阈值':>6}{'误伤(可答)':>12}"
            for name, _ in subgroups:
                header += f"{name:>13}"
            print(header)
            grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60] if field == "gate_rr" else [
                0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65
            ]
            for t in grid:
                row = f"    {t:>6.2f}{sum(1 for v in ans_vals if v < t) / len(ans_vals):>11.1%}"
                for _, members in subgroups:
                    vals = [g[field] for g in members]
                    row += f"{sum(1 for v in vals if v < t) / len(vals):>12.1%} "
                print(row)

            aucs = []
            for name, members in subgroups:
                a = _auc([g[field] for g in members], ans_vals)
                aucs.append(f"{name} {a:.2f}(n={len(members)})" if a is not None else f"{name} n/a")
            print(f"    AUC（1.0 完全可分 / 0.5 瞎猜）：" + " ｜ ".join(aucs))

        print("  （判据方向：score < 阈值 → 判为「语料里没有」。）")
        # 这条提示只在真有 near 分组时才成立。语料扩容后 near 被补成了可答题，
        # 分组里就只剩 far —— 此时还在说「看 near 那一列」会让人去找一个不存在的列。
        names = [name for name, _ in subgroups]
        if any("near" in n for n in names):
            print("  absent·near 是「领域内术语但语料没讲」，比 absent·far 难得多 ——")
            print("  闸门真实可用性由 near 那一列决定，不是 far。")
        else:
            print("  注意：这一版标注里**没有 near 分组**（领域内的缺口已被补成可答题）。")
            print("  于是「闸门到底可不可以用」由 mentioned 那一列决定 —— 它才是难的那一半。")

    if manual_items:
        print(f"\n人工判定类 {len(manual_items)} 条（语料有相关内容但不完全回答，"
              f"不计入 hit@5 也不计入闸门）：")
        for r in manual_items:
            print(f"  {r['q']}　余弦 {r['gate_cos']:.4f} 重排 {r['gate_rr']:.4f}")
            print(f"    依据：{r['note']}")
        print("  （这一类正是「换个方向回答」该服务的对象：既不该硬答，也不该干巴巴说不知道）")

    if gate_items:
        print(f"\n闸门类 {len(gate_items)} 条逐条明细（按重排分数升序）：")
        for row in sorted(gate_items, key=lambda x: x["gate_rr"]):
            tag = f"{row['kind']}" + (f"·{row['note']}" if row.get("note") else "")
            flag = "余弦拦" if row["gate_cos"] < config.min_relevance else "余弦漏"
            print(
                f"  [{tag:<13}] 余弦 {row['gate_cos']:.4f}  重排 {row['gate_rr']:.4f}"
                f"  {flag}  {row['q']}"
            )

    if args.json:
        out = Path(bootstrap.artifact_path("last_report", skill_name, ".json", str(ROOT / "eval")))
        out.write_text(
            json.dumps(
                {
                    "skill": skill_name,
                    "top_n": TOP_N,
                    "per_kind": {
                        k: {
                            "n": v["n"],
                            "hit_rate": {m: v["hit"][m] / v["n"] for m in modes} if v["n"] else {},
                            "cover_at_5": (
                                {m: v["cover"][m] / v["n"] for m in modes} if v["n"] else {}
                            ),
                        }
                        for k, v in per_kind.items()
                    },
                    "mean_overlap": sum(overlaps) / len(overlaps),
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
