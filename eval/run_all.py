"""一条命令的回归门槛：跑评测 → 与基线比 → PASS / FAIL。

为什么需要
----------
改了检索参数或 prompt 之后，只有靠记忆判断有没有退化。这个脚本把它变成一条命令。

    python eval/run_all.py --update-baseline   # 把当前结果立成基线（改完确认没退化就跑一次）
    python eval/run_all.py                     # 跑评测并与基线比，退化就 FAIL（退出码 1）

覆盖三层，**全都不花 LLM 钱**：
  1. 检索层   `run_eval.py --json`          （hit@5 / 覆盖率 / 闸门阈值）
  2. Agent 对照 `run_agent_eval.py --json`   （脚本策略，不调 LLM）
  3. 生成层确定性指标 `run_gen_eval.py --rescore`（复用已保存答案，不改一个字就重算）

口径保护（重要）
----------------
基线里存了 `golden.jsonl` 的**指纹**。标注一改，指标本来就会合法地变 ——
这时对比毫无意义，脚本会直接告诉你去重新立基线，而不是报一个看着很吓人的假 FAIL。
同理语料指纹变了（加/删文档）也会提示。

没有 `--skip-run` 时它会真的重跑前两层（GPU 下检索约 2.4 分钟、Agent 约 1.0 分钟）。
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
sys.path.insert(0, str(ROOT))

import bootstrap  # noqa: E402

# ⚠️ 模块级**不做** chdir / bootstrap.prepare() —— 否则测试一 import 就改变整个进程状态。
# 与 service.py 同一个教训：副作用放进 main()。

SKILL = os.getenv("EVAL_SKILL", "notes")
BASELINE_DIR = EVAL_DIR / "baseline"

# 会影响所有指标的配置项。基线里记它们的**生效值**，对比时先比配置再比指标 ——
# 否则会出现这种坑：上一次跑的是「关掉重排」的实验，报告留在磁盘上，
# 然后 --update-baseline 把实验配置立成了基线，之后怎么比都 PASS（实测踩过）。
CONFIG_KEYS = (
    "rerank_enabled",
    "rerank_candidates",
    "rerank_device",
    "top_k",
    "min_relevance",
    "enable_rewrite",
    "embedding_model",
    "context_max_tokens",
)


def _effective_config() -> dict:
    """读取当前**生效**的检索配置（含环境变量覆盖）。只在 main() 里调用。"""
    from dotenv import load_dotenv

    load_dotenv(bootstrap.env_file_for(SKILL))
    from rag_core import RAGConfig

    config = RAGConfig()
    return {k: getattr(config, k) for k in CONFIG_KEYS}


def config_changes(base: dict, cur: dict) -> list:
    """返回 [(键, 基线值, 当前值)] —— 只看两边都有的键。纯函数。"""
    return [
        (k, base[k], cur[k])
        for k in sorted(set(base) & set(cur))
        if base[k] != cur[k]
    ]


def stale_baseline_files(existing: list, keep: list) -> list:
    """基线目录里**该删**的层文件：存在、但不在本次基线记录里。纯函数，便于单测。

    不删的后果：下次跑门禁会拿**上一版的报告**当基线比，报出一堆假退化。
    这个坑踩过两次 —— 一次是拿了 `RERANK_ENABLED=false` 的遗留报告立基线，
    一次是生成层报告被归档后、旧的 `gen.*.json` 还留在基线目录里。
    """
    return sorted(set(existing) - set(keep))


def needs_gen_rescore(no_gen: bool, report_exists: bool) -> bool:
    """该不该跑生成层的 --rescore。纯函数，便于单测。

    **缺报告时不能跑**：`run_gen_eval.py --rescore` 会以退出码 1 结束，
    于是整个门禁报 FAIL —— 但那不是「指标退化」，是这一层没有数据。
    缺数据应该**跳过并说明**，不该伪装成失败。
    """
    return (not no_gen) and report_exists


def deltas(base: dict, cur: dict, tol: float):
    """比较两套扁平指标。返回 (全部行, 退化行)。纯函数，便于单测。

    只报**退化**（差值 < -tol）；提升不拦，但会标出来。
    """
    rows, bad = [], []
    for key in sorted(set(base) | set(cur)):
        b, c = base.get(key), cur.get(key)
        if b is None or c is None:
            rows.append((key, b, c, None))
            continue
        d = c - b
        rows.append((key, b, c, d))
        if d < -tol:
            bad.append((key, b, c, d))
    return rows, bad


def _p(name: str, ext: str) -> Path:
    return Path(bootstrap.artifact_path(name, SKILL, ext, str(EVAL_DIR)))


def _fingerprint(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]


def _run(script: str, extra=()) -> None:
    """跑子脚本。输出重定向到文件 —— 不用管道（受限环境里管道可能被拒）。"""
    log = EVAL_DIR / f"_run_all_{Path(script).stem}.log"
    cmd = [sys.executable, "-u", str(EVAL_DIR / script), *extra]
    print(f"  → {' '.join(Path(c).name if i else c for i, c in enumerate(cmd))}")
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        rc = subprocess.call(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
    if rc != 0:
        raise SystemExit(f"{script} 退出码 {rc}，日志：{log}")


# ---------- 从三份报告里抽出可比的扁平指标 ----------

def _flat_retrieval(data: dict) -> dict:
    out = {}
    for kind, bucket in (data.get("per_kind") or {}).items():
        for arm, v in (bucket.get("hit_rate") or {}).items():
            out[f"{kind}/{arm}"] = v
    for kind, bucket in (data.get("per_kind") or {}).items():
        for arm, v in (bucket.get("cover_at_5") or {}).items():
            out[f"cover:{kind}/{arm}"] = v
    if "mean_overlap" in data:
        out["(诊断) 两路平均重叠率"] = data["mean_overlap"]
    return out


def _flat_agent(data: dict) -> dict:
    out = {}
    for kind, bucket in (data.get("per_kind") or {}).items():
        for arm, v in (bucket.get("hit_rate") or {}).items():
            out[f"{kind}/{arm}"] = v
    return out


def _flat_gen(data: dict) -> dict:
    out = {}
    for arm, s in (data.get("stats") or {}).items():
        n = s.get("n_ans") or 0
        g = s.get("n_gate") or 0
        m = s.get("n_manual") or 0
        if n:
            out[f"{arm}: 引对率"] = s["cited_right"] / n
            out[f"{arm}: 上下文命中"] = s["context_ok"] / n
        if g:
            out[f"{arm}: 闸门声明缺口"] = s["declares_gap"] / g
        if m:
            out[f"{arm}: partial 给方向"] = s["manual_cited"] / m
        cov = s.get("cov") or []
        if cov:
            out[f"{arm}: 要点覆盖率"] = sum(cov) / len(cov)
    return out


def _print_diff(title: str, base: dict, cur: dict, tol: float) -> bool:
    print(f"\n【{title}】")
    if not base:
        print("  基线里没有这一层，跳过（用 --update-baseline 补上）")
        return True
    rows, bad = deltas(base, cur, tol)
    print(f"  {'指标':<34}{'基线':>9}{'当前':>9}{'差值':>9}")
    for k, b, c, d in rows:
        mark = ""
        if d is None:
            mark = "  ← 基线/当前缺一边"
        elif d < -tol:
            mark = "  ← 退化"
        elif d > tol:
            mark = "  ← 提升"
        bs = "—" if b is None else f"{b:.1%}"
        cs = "—" if c is None else f"{c:.1%}"
        ds = "" if d is None else f"{d:+.1%}"
        print(f"  {k:<34}{bs:>9}{cs:>9}{ds:>9}{mark}")
    if bad:
        print(f"\n  退化 {len(bad)} 项（阈值 {tol:.0%}）：")
        for k, b, c, d in bad:
            print(f"    {k}: {b:.1%} → {c:.1%} ({d:+.1%})")
        return False
    print(f"\n  没有超过阈值（{tol:.0%}）的退化 ✓")
    return True


def main() -> None:
    os.chdir(ROOT)
    bootstrap.prepare()

    parser = argparse.ArgumentParser()
    parser.add_argument("--update-baseline", action="store_true", help="把当前结果存成基线")
    parser.add_argument("--tol", type=float, default=0.02, help="允许的退化幅度（默认 2 个百分点）")
    parser.add_argument("--skip-run", action="store_true", help="不重跑，只比较已有报告")
    parser.add_argument("--no-gen", action="store_true", help="跳过生成层确定性指标（--rescore）")
    args = parser.parse_args()

    golden = _p("golden", ".jsonl")
    cur_fp = _fingerprint(golden)
    cur_cfg = _effective_config()
    print(f"skill={SKILL}  标注指纹={cur_fp}  基线目录={BASELINE_DIR}")
    print(f"生效配置：{cur_cfg}")

    if args.update_baseline:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        meta = {
            "skill": SKILL,
            "golden_fingerprint": cur_fp,
            "config": cur_cfg,
            "reports": {},
        }
        for name, ext, path in (
            ("retrieval", ".json", _p("last_report", ".json")),
            ("agent", ".json", _p("last_agent_report", ".json")),
            ("gen", ".json", _p("last_gen_report", ".json")),
        ):
            if not path.exists():
                print(f"  ⚠ 缺 {path.name}，这一层不进基线（先跑一次对应评测）")
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            (BASELINE_DIR / f"{name}.{SKILL}.json").write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )
            meta["reports"][name] = path.name
            print(f"  {name:<10} ← {path.name}")
        # 这一版没进基线的层，要把**上一版留下的**文件删掉：留着它，
        # 下次跑门禁会拿那份旧报告当基线比，报出一堆假退化（踩过两次）
        present = [p.name for p in BASELINE_DIR.glob(f"*.{SKILL}.json")]
        keep = [f"{n}.{SKILL}.json" for n in meta["reports"]] + [f"meta.{SKILL}.json"]
        for stale in stale_baseline_files(present, keep):
            (BASELINE_DIR / stale).unlink()
            print(f"  🗑 删除上一版残留的基线文件：{stale}（这一层本轮没进基线）")
        (BASELINE_DIR / f"meta.{SKILL}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            "\n⚠ 基线记的是**磁盘上现有的报告**。如果你上一次跑的是实验配置"
            "（比如 RERANK_ENABLED=false），那份报告会被立成基线。\n"
            "  保险做法：先在正常配置下重跑一遍评测，再 --update-baseline。"
        )
        return

    meta_path = BASELINE_DIR / f"meta.{SKILL}.json"
    if not meta_path.exists():
        raise SystemExit(
            f"还没有基线：{meta_path}\n先跑一次：python eval\\run_all.py --update-baseline"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # **口径保护**：标注变了指标会合法地变，这时硬比会报假的 FAIL
    if meta.get("golden_fingerprint") != cur_fp:
        print(
            f"\n⚠ 标注指纹变了（基线 {meta.get('golden_fingerprint')} → 现在 {cur_fp}）。\n"
            "  改了 seeds/golden 之后，指标本来就会变，此时和旧基线对比**没有意义**。\n"
            "  确认新标注没问题后重新立基线：python eval\\run_all.py --update-baseline"
        )
        return

    if not args.skip_run:
        print("\n跑检索层评测…")
        _run("run_eval.py", ["--json"])
        print("跑 Agent 对照评测…")
        _run("run_agent_eval.py", ["--json"])
    if not args.no_gen:
        gen_report = _p("last_gen_report", ".json")
        if not needs_gen_rescore(args.no_gen, gen_report.exists()):
            # **缺报告就不要硬跑**：子脚本会以退出码 1 结束，整个门禁跟着"失败"，
            # 但那不是「有指标退化」—— 是这一层根本没数据。
            # （踩过：把 week16 之前的旧生成报告归档之后，--skip-run 直接 FAIL）
            print(
                f"⚠ 没有 {gen_report.name}，生成层跳过（它也不在基线里，不参与比较）。\n"
                "   要恢复这一层：跑一次生成 python eval\\run_gen_eval.py"
            )
        else:
            print("重算生成层确定性指标（复用已保存答案，不调 LLM）…")
            _run("run_gen_eval.py", ["--rescore"])

    # **配置保护**：配置变了指标本来就会变，先说清楚，否则你会把配置差异当成代码退化
    cfg_diff = config_changes(meta.get("config") or {}, cur_cfg)
    if cfg_diff:
        print("\n⚠ 生效配置和基线不一致 —— 下面的指标差异可能只是配置造成的，不一定是代码退化：")
        for k, b, c in cfg_diff:
            print(f"    {k}: {b} → {c}")

    def load(name, path):
        base_path = BASELINE_DIR / f"{name}.{SKILL}.json"
        base = json.loads(base_path.read_text(encoding="utf-8")) if base_path.exists() else {}
        cur = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return base, cur

    ok = True
    layers = [
        ("检索层", "retrieval", _p("last_report", ".json"), _flat_retrieval),
        ("Agent 对照", "agent", _p("last_agent_report", ".json"), _flat_agent),
        ("生成层（确定性）", "gen", _p("last_gen_report", ".json"), _flat_gen),
    ]
    for title, name, path, flat in layers:
        if not path.exists():
            print(f"\n【{title}】报告不存在，跳过：{path.name}")
            continue
        base, cur = load(name, path)
        ok &= _print_diff(title, flat(base), flat(cur), args.tol)

    print("\n" + "=" * 60)
    print("结论：PASS ✓" if ok else "结论：FAIL ✗ —— 有指标退化，看上面标「退化」的行")
    print("（输出日志在 eval\\_run_all_*.log）")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
