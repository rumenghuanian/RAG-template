"""把人工复核结果与裁判判定对比，算出「裁判 vs 人工」一致率。

这是项目里唯一一个**用人工判断校准自动判分**的地方。没有它，"忠实度 97.7%"
只是一个自称；有了它，才知道这个数字的误差方向（裁判偏宽还是偏严）。

用法：
    python eval/score_human_review.py                      # 读 eval/human_labels.json
    python eval/score_human_review.py --labels eval/xxx.json

判读口径（写死在这里，避免事后挑好看的算法）：
- **一致率** = 人工与裁判判定相同的条数 / 双方都有明确判定的条数
- 「拿不准」的人工标注**不计入**分母（它没有表态），但要单独报出条数
- 裁判判定失败（faithful=None）的条目不参与
- 分别报 **裁判偏宽**（人说不忠实、裁判说忠实）与 **裁判偏严**（反之）——方向比总数重要
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bootstrap  # noqa: E402

bootstrap.prepare()

import run_gen_eval as g  # noqa: E402

HUMAN_TO_BOOL = {"忠实": True, "不忠实": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="", help="人工标注 json（默认 eval/human_labels.json）")
    parser.add_argument("--arm", default="single")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    labels_path = Path(args.labels) if args.labels else Path(
        bootstrap.artifact_path("human_labels", g.SKILL, ".json", g.EVAL_DIR)
    )
    if not labels_path.exists():
        raise SystemExit(
            f"找不到人工标注：{labels_path}\n"
            "先用 eval/build_human_review.py 生成复核表，标完点「导出结果」，"
            "把 human_labels.json 放回 eval/ 目录。"
        )
    human = json.loads(labels_path.read_text(encoding="utf-8"))
    if human.get("seed") is not None:
        print(f"人工标注：{labels_path.name}  种子={human['seed']}  "
              f"（复核表指纹 {human.get('fingerprint')}）")

    report = Path(args.report) if args.report else g.REPORT
    data = json.loads(report.read_text(encoding="utf-8"))

    # 重建同一套 id（与 build_human_review 一致：按报告顺序编号）
    judge = {}
    for i, row in enumerate(data["detail"], 1):
        arm = (row.get("arms") or {}).get(args.arm)
        if not arm:
            continue
        verdict = (arm.get("judge") or {}).get("faithful")
        judge[f"{args.arm}#{i:02d}"] = {
            "faithful": verdict,
            "reason": (arm.get("judge") or {}).get("reason", ""),
            "question": row["question"],
            "kind": row["kind"],
        }

    if human.get("fingerprint") and human["fingerprint"] != _fp(report):
        print(
            f"\n⚠ 复核表指纹（{human['fingerprint']}）与当前报告（{_fp(report)}）不一致 ——\n"
            "  你可能换了报告却没重新生成复核表，下面的对比**不可信**，请先重建复核表。\n"
        )

    rows, skipped_unsure, skipped_nojudge, missing = [], 0, 0, 0
    for rid, info in sorted(judge.items()):
        got = (human.get("labels") or {}).get(rid) or {}
        label = (got.get("label") or "").strip()
        if not label:
            missing += 1
            continue
        if label == "拿不准":
            skipped_unsure += 1
            continue
        if info["faithful"] is None:
            skipped_nojudge += 1
            continue
        rows.append(
            {
                "id": rid,
                "question": info["question"],
                "kind": info["kind"],
                "human": HUMAN_TO_BOOL[label],
                "judge": bool(info["faithful"]),
                "note": (got.get("note") or "").strip(),
                "reason": info["reason"],
            }
        )

    print(
        f"\n可对比 {len(rows)} 条"
        f"（人工没标的 {missing} 条、「拿不准」{skipped_unsure} 条、"
        f"裁判未判定的 {skipped_nojudge} 条不参与）"
    )
    if not rows:
        raise SystemExit("没有可对比的条目 —— 先完成人工标注")

    agree = [r for r in rows if r["human"] == r["judge"]]
    fp = [r for r in rows if r["human"] is False and r["judge"] is True]   # 裁判偏宽
    fn = [r for r in rows if r["human"] is True and r["judge"] is False]   # 裁判偏严

    print(f"\n【一致率】{len(agree)}/{len(rows)} = {len(agree) / len(rows):.1%}")
    print(f"  裁判偏宽（人说不忠实、裁判说忠实）：{len(fp)} 条")
    print(f"  裁判偏严（人说忠实、裁判说不忠实）：{len(fn)} 条")

    print("\n【混淆矩阵】行=人工，列=裁判")
    print(f"  {'':<8}{'裁判忠实':>10}{'裁判不忠实':>12}")
    for hv, name in ((True, "人工忠实"), (False, "人工不忠实")):
        a = sum(1 for r in rows if r["human"] is hv and r["judge"] is True)
        b = sum(1 for r in rows if r["human"] is hv and r["judge"] is False)
        print(f"  {name:<8}{a:>10}{b:>12}")

    print("\n【按题型】")
    for kind in sorted({r["kind"] for r in rows}):
        g2 = [r for r in rows if r["kind"] == kind]
        ag = sum(1 for r in g2 if r["human"] == r["judge"])
        print(f"  {kind:<12}{ag}/{len(g2)} = {ag / len(g2):.0%}")

    if fp or fn:
        print("\n【分歧明细】（这是最该看的部分：裁判错在哪里）")
        for r in fp + fn:
            direction = "裁判偏宽" if r in fp else "裁判偏严"
            print(f"\n  [{direction}] {r['id']} [{r['kind']}] {r['question']}")
            if r["note"]:
                print(f"    人工备注：{r['note']}")
            print(f"    裁判理由：{str(r['reason'])[:150]}")
    else:
        print("\n没有分歧 ✓")

    print(f"\n（人工判定分布：{dict(Counter('忠实' if r['human'] else '不忠实' for r in rows))}）")


def _fp(path: Path) -> str:
    import hashlib

    return hashlib.md5(path.read_bytes()).hexdigest()[:12]


if __name__ == "__main__":
    main()
