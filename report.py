"""python report.py -> out/report.html from out/results.json (run bench.py first)."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

HERE = Path(__file__).parent
ARC = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"]
CSS = """
:root{--bg:#f8f8f6;--fg:#1b1b19;--mut:#6c6a64;--line:#e3e1db;--card:#fff;--ok:#1d7a4f;--acc:#2f62c9}
@media (prefers-color-scheme:dark){:root{--bg:#121211;--fg:#ecebe6;--mut:#9a978f;--line:#2b2a27;--card:#1a1a18;--ok:#58c08d;--acc:#7fa6ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:30px 16px}h1{font-size:25px;margin:0}h2{font-size:17px;margin:28px 0 10px}.mut{color:var(--mut)}
.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line)}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}th{font-size:11px;text-transform:uppercase;color:var(--mut)}
.num{font-variant-numeric:tabular-nums;text-align:right}.ok{color:var(--ok);font-weight:600}code{font:12px ui-monospace,monospace}
.tasks{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px}
.t{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}.t svg{vertical-align:middle}
"""


def grid_svg(g, max_px=110):
    h, w = len(g), len(g[0])
    cell = min(9, max_px / max(h, w))
    rects = "".join(f'<rect x="{x * cell}" y="{y * cell}" width="{cell}" height="{cell}" fill="{ARC[v]}"/>'
                    for y, row in enumerate(g) for x, v in enumerate(row))
    return f'<svg width="{w * cell}" height="{h * cell}" style="outline:1px solid #555">{rects}</svg>'


def main():
    data = json.loads((HERE / "out" / "results.json").read_text())
    s = data["summary"]
    for k, v in s["splits"].items():
        both = [r for r in data["tasks"] if r["set"] == k and r["blind"]["test"] is not None and r["guided"]["test"] is not None]
        v["sum_ratio"] = round(sum(r["blind"]["evals"] for r in both) / max(1, sum(r["guided"]["evals"] for r in both)), 2)
        v["guided_worse"] = sum(1 for r in both if r["guided"]["evals"] > r["blind"]["evals"])
    rows = "".join(
        f"<tr><td>{k}</td><td class=num>{v['tasks']}</td><td class=num>{v['blind']['test_correct']}</td>"
        f"<td class='num ok'>{v['guided']['test_correct']}</td><td class=num>{v.get('blind_median_evals', 0):,.0f}</td>"
        f"<td class='num ok'>{v.get('guided_median_evals', 0):,.0f}</td><td class='num ok'>{v.get('median_speedup', '-')}×</td><td class=num>{v['sum_ratio']}×</td><td class=num>{v['guided_worse']}</td></tr>"
        for k, v in s["splits"].items())
    solved = [r for r in data["tasks"] if r["guided"]["test"]]
    solved.sort(key=lambda r: -(r["blind"]["evals"] / r["guided"]["evals"]))
    cards = ""
    for r in solved[:24]:
        task = json.loads((HERE / "data" / "arc" / r["set"] / f"{r['task']}.json").read_text())
        ex = task["train"][0]
        only = " · <b class=ok>guided only</b>" if not r["blind"]["test"] else ""
        cards += (f"<div class=t><div><code>{r['task']}</code> <span class=mut>{r['set']}</span>{only}</div>"
                  f"<div style='margin:8px 0'>{grid_svg(ex['input'])} <span class=mut>→</span> {grid_svg(ex['output'])}</div>"
                  f"<div><code>{escape(r['guided']['program'])}</code></div>"
                  f"<div class=mut>evals: blind {r['blind']['evals']:,} · guided <b class=ok>{r['guided']['evals']:,}</b></div></div>")
    html = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Guided ARC Search</title><style>{CSS}</style></head><body><main>
<h1>Learned guide for ARC program search</h1>
<div class=mut>{s['primitives']} primitives · programs up to depth {s['max_depth']} · budget {s['budget']:,} programs per task · public ARC-AGI-1 tasks · unofficial prototype by Kushagra Pratap Singh</div>
<h2>Blind vs guided (solved = all train pairs consistent and the hidden test output exactly right)</h2>
<div class=scroll><table><tr><th>Split</th><th class=num>Tasks</th><th class=num>Blind solved</th><th class=num>Guided solved</th>
<th class=num>Blind median evals</th><th class=num>Guided median evals</th><th class=num>Median speed-up</th><th class=num>Total-evals ratio</th><th class=num>Tasks guide was slower</th></tr>{rows}</table></div>
<p class=mut>Median and total columns are over tasks where both searches found a program. The median saving is large because most solvable tasks are one-step and the guide ranks the right primitive near the top; the total ratio is small because on a handful of tasks the guide ranks the right program <i>lower</i> than breadth-first order — that is the honest headroom. The guide is an MLP trained only on tasks generated from the DSL itself; it never saw ARC.</p>
<h2>Solved tasks, biggest search savings first</h2><div class=tasks>{cards}</div>
</main></body></html>"""
    (HERE / "out" / "report.html").write_text(html, encoding="utf-8")
    print("wrote out/report.html")


if __name__ == "__main__":
    main()
