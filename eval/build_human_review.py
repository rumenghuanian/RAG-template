"""生成「人工复核表」，用来给裁判做人工基准（judge vs human 一致率）。

为什么需要它
------------
裁判已经过了两个方向的校准（打乱上下文抓「恒点头」、注入假细节抓「恒摇头」），
但那只证明它**不是橡皮图章**，不证明它**准**。官方数据是 LLM 判官与人工一致率
约 75%~87% —— 所以"忠实度 97.7%"这个数必须配一个人工基准才有意义。

三条必须守的规矩（否则标出来的数没有价值）
------------------------------------------
1. **不显示裁判结论**：显示了就会锚定，人只会去"确认"裁判的判断。
2. **给模型真正读到的正文**（截断后的那版）：给全文等于让人判一个模型没见过的题。
3. **明写判据**并给「拿不准」这个选项：逼人在模糊处表态，而不是猜或者跳过。

用法：
    python eval/build_human_review.py            # 生成 eval/human_review.html
    python eval/build_human_review.py --arm agent
默认只做有裁判判定的那些题（可答题），按固定种子打乱顺序（避免按类型连着标产生惯性）。
"""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bootstrap  # noqa: E402

bootstrap.prepare()

import run_gen_eval as g  # noqa: E402

SEED = 20260922

CRITERIA = [
    ("忠实", "与上下文不矛盾；只是措辞不同、做了合理概括、或明确说明「这里没有」都算忠实。"),
    ("不忠实", "与上下文矛盾，**或**出现了上下文里没有的**具体**断言（数字、名称、步骤、结论、出处）。"),
    ("拿不准", "上下文本身被截断、或你无法判断某句是否有依据 —— 选这个，并在备注里写一句为什么。"),
]

HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>人工复核 · 裁判基准</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
         max-width: 1000px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .muted {{ opacity: .7; font-size: 13px; }}
  .crit {{ background: #f6f8fa; border-left: 3px solid #0969da; padding: 10px 14px;
          border-radius: 6px; margin: 14px 0; font-size: 14px; }}
  .bar {{ position: sticky; top: 0; background: canvas; padding: 10px 0; z-index: 5;
         border-bottom: 1px solid #d0d7de; display: flex; gap: 12px; align-items: center; }}
  .bar b {{ font-variant-numeric: tabular-nums; }}
  button {{ font: inherit; padding: 8px 14px; border-radius: 6px; cursor: pointer;
           border: 1px solid #d0d7de; background: canvas; }}
  button:hover {{ background: #f3f4f6; }}
  .pick button {{ margin-right: 8px; }}
  .pick button.on {{ outline: 2px solid #0969da; font-weight: 600; }}
  .card {{ border: 1px solid #d0d7de; border-radius: 8px; padding: 16px 18px; margin: 14px 0; }}
  .q {{ font-weight: 600; font-size: 16px; margin-bottom: 8px; }}
  .ans {{ white-space: pre-wrap; background: #fbfcfd; border-radius: 6px;
         padding: 10px 12px; margin: 8px 0; }}
  details {{ margin-top: 8px; }}
  summary {{ cursor: pointer; color: #0969da; font-size: 14px; }}
  .ctx {{ white-space: pre-wrap; font-size: 13px; opacity: .85; max-height: 340px;
         overflow: auto; background: #f6f8fa; padding: 10px; border-radius: 6px; margin-top: 8px; }}
  input.note {{ width: 100%; padding: 7px 9px; margin-top: 10px; box-sizing: border-box;
               border: 1px solid #d0d7de; border-radius: 6px; }}
  .done {{ opacity: .55; }}
  .kind {{ font-size: 12px; opacity: .6; font-weight: 400; }}
</style>
</head>
<body>
<h1>人工复核 · 给裁判做基准</h1>
<p class="muted">
  共 <b id="total"></b> 条。判据见下。<b>页面不显示任何自动判分结果</b>，避免锚定。
  进度自动存在浏览器里，可以关掉页面明天接着标。标完点右上角「导出」。
</p>
<div class="crit">
  {criteria_html}
  <div style="margin-top:6px" class="muted">
    提示：答案通常是原文的概括，<b>不必逐句回读上下文</b>；只在觉得某句可疑时展开上下文核对。
    这一条本身就是"筛查"，不是逐字审计 —— 这点我会写进 README，不夸大。
  </div>
</div>
<div class="bar">
  <b><span id="done">0</span>/<span id="total2"></span></b>
  <button onclick="jump(-1)">← 上一条</button>
  <button onclick="jump(1)">下一条 →</button>
  <span class="muted">快捷键：1 忠实 / 2 不忠实 / 3 拿不准</span>
  <span id="persist" class="muted"></span>
  <button style="margin-left:auto" onclick="exportJSON()">导出结果</button>
</div>
<div id="resume"></div>
<div id="app"></div>

<script>
const ITEMS = {items_json};
const KEY = "human_review_" + {fingerprint_json};

// localStorage 在 file:// 下可能被浏览器拒绝（SecurityError）。
// 直接调用会让整个 render 抛异常 → **白屏**。所以包一层兜底：
// 存不了就退化成"只在内存里"，页面照常能标，只是关掉会丢进度。
let store = (() => {{
  try {{
    localStorage.setItem("__probe__", "1");
    localStorage.removeItem("__probe__");
    return localStorage;
  }} catch (e) {{
    const mem = {{}};
    return {{
      getItem: k => (k in mem ? mem[k] : null),
      setItem: (k, v) => {{ mem[k] = String(v); }},
      _memoryOnly: true
    }};
  }}
}})();
let labels = JSON.parse(store.getItem(KEY) || "{{}}");
const byId = Object.fromEntries(ITEMS.map(i => [i.id, i]));

function save() {{ try {{ store.setItem(KEY, JSON.stringify(labels)); }} catch (e) {{}} }}

function render() {{
  const app = document.getElementById("app");
  app.innerHTML = "";
  ITEMS.forEach((it, idx) => {{
    const card = document.createElement("div");
    card.className = "card" + (labels[it.id] ? " done" : "");
    card.id = "card-" + it.id;
    const cur = labels[it.id] || {{label: "", note: ""}};
    card.innerHTML = `
      <div class="q">${{idx + 1}}. ${{esc(it.question)}} <span class="kind">#${{it.id}}</span></div>
      <div class="ans">${{esc(it.answer)}}</div>
      <details><summary>展开模型实际读到的上下文（${{it.context_chars}} 字，已按 token 预算截断）</summary>
        <div class="ctx">${{esc(it.context)}}</div>
      </details>
      <div class="pick" style="margin-top:10px">
        ${{["忠实", "不忠实", "拿不准"].map(l =>
            `<button data-id="${{it.id}}" data-l="${{l}}"
              class="${{cur.label === l ? "on" : ""}}">${{l}}</button>`).join("")}}
      </div>
      <input class="note" placeholder="备注（选「不忠实」时请写是哪一句没依据；选「拿不准」写为什么）"
             data-note="${{it.id}}" value="${{esc(cur.note || "")}}">
    `;
    app.appendChild(card);
  }});
  app.querySelectorAll("button[data-l]").forEach(b => b.onclick = () => {{
    const id = b.dataset.id;
    labels[id] = labels[id] || {{label: "", note: ""}};
    labels[id].label = b.dataset.l;
    save(); render(); count();
    const el = document.getElementById("card-" + id);
    if (el) el.scrollIntoView({{behavior: "smooth", block: "start"}});
  }});
  app.querySelectorAll("input[data-note]").forEach(inp => inp.oninput = () => {{
    const id = inp.dataset.note;
    labels[id] = labels[id] || {{label: "", note: ""}};
    labels[id].note = inp.value;
    save();
  }});
  count();
}}

function esc(s) {{
  return String(s).replace(/[&<>"']/g, c => (
    {{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
}}

function count() {{
  const n = ITEMS.filter(i => labels[i.id] && labels[i.id].label).length;
  document.getElementById("done").textContent = n;
  document.getElementById("total2").textContent = ITEMS.length;
  const hint = document.getElementById("resume");
  if (n > 0 && n < ITEMS.length) {{
    hint.innerHTML = `<p class="muted">已标 ${{n}} 条。
      <a href="#" onclick="jump(0);return false;">跳到第一条未标的</a></p>`;
  }} else if (n === ITEMS.length) {{
    hint.innerHTML = `<p class="muted">全部标完 ✓ 点右上角「导出结果」。</p>`;
  }} else {{
    hint.innerHTML = "";
  }}
}}

function jump(d) {{
  if (d === 0) {{   // 0 = 跳到第一条未标
    const first = ITEMS.find(i => !(labels[i.id] && labels[i.id].label));
    const el = document.getElementById("card-" + (first ? first.id : ITEMS[0].id));
    if (el) el.scrollIntoView({{behavior: "smooth", block: "start"}});
    return;
  }}
  const el = document.getElementById("card-" + ITEMS[0].id);
  if (el) el.scrollIntoView({{behavior: d < 0 ? "start" : "end"}});
}}

document.addEventListener("keydown", e => {{
  const map = {{"1": "忠实", "2": "不忠实", "3": "拿不准"}};
  if (!map[e.key]) return;
  if (e.target.tagName === "INPUT") return;
  const first = ITEMS.find(i => !(labels[i.id] && labels[i.id].label));
  if (!first) return;
  labels[first.id] = labels[first.id] || {{label: "", note: ""}};
  labels[first.id].label = map[e.key];
  save(); render();
}});

function exportJSON() {{
  const out = {{
    fingerprint: {fingerprint_json},
    seed: {seed},
    judged_ok: {judged_ok},
    labels: Object.fromEntries(ITEMS.map(i => [i.id,
      labels[i.id] || {{label: "", note: ""}}]))
  }};
  const blob = new Blob([JSON.stringify(out, null, 2)], {{type: "application/json"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "human_labels.json";
  a.click();
  alert("已下载 human_labels.json —— 把它放回项目的 eval/ 目录，然后跑：\\n"
      + "python eval/score_human_review.py");
}}

document.getElementById("total").textContent = ITEMS.length;
if (store._memoryOnly) {{
  document.getElementById("persist").textContent =
    "⚠ 此浏览器不允许本地存储，进度不会保留（别关页面）";
}}
if (ITEMS.length && labels[ITEMS[0].id]) {{ /* 有历史进度时由 count() 给出跳转提示 */ }}
render();
</script>
</body>
</html>
"""


def _fingerprint(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="single", help="哪一臂（默认 single）")
    parser.add_argument("--report", default="", help="指定报告文件（默认当前 skill 的生成层报告）")
    parser.add_argument("--seed", type=int, default=SEED, help="打乱顺序的种子")
    args = parser.parse_args()

    report = Path(args.report) if args.report else g.REPORT
    if not report.exists():
        raise SystemExit(f"找不到报告：{report}\n先跑：python eval/run_gen_eval.py --arms {args.arm}")

    data = json.loads(report.read_text(encoding="utf-8"))
    items = []
    for i, row in enumerate(data["detail"], 1):
        arm = (row.get("arms") or {}).get(args.arm)
        if not arm:
            continue
        verdict = arm.get("judge") or {}
        if verdict.get("faithful") is None:
            continue          # 只做有裁判判定的题，否则没法比
        ctx = g._fit_reference(arm.get("context_texts") or [])
        items.append(
            {
                "id": f"{args.arm}#{i:02d}",
                "question": row["question"],
                "answer": arm.get("answer") or "",
                "context": ctx,
                "context_chars": len(ctx),
            }
        )

    if not items:
        raise SystemExit(f"报告里没有带裁判判定的 {args.arm} 臂答案（先跑 --judge-only）")

    # 固定种子打乱：避免按题型连着标产生惯性，同时保证可复现
    rng = random.Random(args.seed)
    rng.shuffle(items)

    criteria_html = "".join(
        f"<div><b>{name}</b>：{desc}</div>" for name, desc in CRITERIA
    )
    html = HTML.format(
        criteria_html=criteria_html,
        items_json=json.dumps(items, ensure_ascii=False).replace("</", "<\\/"),
        fingerprint_json=json.dumps(_fingerprint(report)),
        seed=args.seed,
        judged_ok=len(items),
    )
    out = Path(bootstrap.artifact_path("human_review", g.SKILL, ".html", g.EVAL_DIR))
    out.write_text(html, encoding="utf-8")

    print(f"skill={g.SKILL}  臂={args.arm}  共 {len(items)} 条（已按种子 {args.seed} 打乱）")
    print(f"复核表：{out}")
    print(f"参考报告：{report.name}（指纹 {_fingerprint(report)}）")
    print("\n用法：用浏览器打开上面那个 html → 逐条选「忠实 / 不忠实 / 拿不准」→ 点「导出结果」")
    print("      把下载到的 human_labels.json 放回 eval/ 目录，再跑：")
    print("      python eval/score_human_review.py")
    print("\n⚠️ 页面里**故意不显示**裁判的判定，避免锚定。标完再对比才有意义。")


if __name__ == "__main__":
    main()
