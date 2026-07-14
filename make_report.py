"""
Generate report.html — the living scoreboard for the learned optimizer.

Parses the newest eval_matrix*.txt fairness table and renders a self-contained,
theme-aware HTML page. Rerun after each checkpoint eval; republish to the same
artifact URL to update in place.

Usage: .venv/bin/python make_report.py [--eval eval_matrixN.txt] [--out report.html]
"""
import argparse
import glob
import html
import math
import os
import re
import subprocess
from datetime import datetime, timezone

HOLDOUT_NOTE = {
    "fashion": "dataset held out", "pendigits": "dataset held out",
    "text8": "corpus held out", "h64": "width held out", "(32,32)": "shape held out",
    "B=512": "batch size unseen",
}

MILESTONES = [
    ("v2", "MNIST-only distro; matches tuned Muon on 100-step MNIST probe"),
    ("v4", "task zoo; first zero-shot wins over ALL tuned baselines (pendigits@20)"),
    ("v5", "time inputs + cross-layer + capacity; calibration fixed (lambda*=1)"),
    ("v7", "noise + width variation; beats Muon on MNIST h16@20; wins nanoGPT@100"),
    ("v8", "width support to 96: fashion-h64 probe 46-68% -> 93%"),
    ("v9", "resampled episodes (real SGD), d<=128 nanoGPTs, OpenML-54 population"),
    ("v10", "muP fan-in gauge: zero-shot descends at 10.7M, early parity with Muon"),
    ("v11", "horizon fix: episodes to 2000 steps (v10 plateaued past 1000) - TRAINING"),
]


def parse_eval(path):
    budgets = {}
    budget, task = None, None
    for line in open(path):
        m = re.match(r"=== budget: (\d+) steps ===", line.strip())
        if m:
            budget = int(m.group(1))
            budgets[budget] = []
            continue
        if budget is None:
            continue
        if re.match(r"^  \S", line) and "loss" not in line:
            task = {"name": line.strip(), "rows": []}
            budgets[budget].append(task)
            continue
        m = re.match(r"^    (\S.*?)\s+loss\s+([\d.]+|nan)\s+acc\s+([\d.]+)", line)
        if m and task is not None:
            task["rows"].append((m.group(1).strip(), float(m.group(2)),
                                 float(m.group(3))))
    return budgets


def verdict(task):
    learned = [r for r in task["rows"] if r[0].startswith("learned")]
    base = [r for r in task["rows"] if not r[0].startswith("learned")]
    lz = min((r[1] for r in learned), default=math.inf)
    bb = min(base, key=lambda r: r[1])
    tol = max(1e-4, 0.15 * max(lz, bb[1]))
    if lz < bb[1] - tol:
        return "win", bb
    if lz > bb[1] + tol:
        return "loss", bb
    return "tie", bb


def bar(loss, lo=1e-4, hi=10.0):
    v = min(max(loss, lo), hi)
    frac = (math.log10(hi) - math.log10(v)) / (math.log10(hi) - math.log10(lo))
    return max(2.0, 100.0 * frac)


def fmt(x):
    return f"{x:.4f}" if x < 100 else f"{x:.1f}"


def render(budgets, eval_path, checkpoint, out):
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip() or "?"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tallies = {}
    for b, tasks in budgets.items():
        t = {"win": 0, "tie": 0, "loss": 0}
        for task in tasks:
            t[verdict(task)[0]] += 1
        tallies[b] = t

    def chips(t):
        return (f'<span class="chip win">{t["win"]} win</span>'
                f'<span class="chip tie">{t["tie"]} tie</span>'
                f'<span class="chip loss">{t["loss"]} loss</span>')

    sections = ""
    for b in sorted(budgets):
        rows_html = ""
        for task in budgets[b]:
            v, bb = verdict(task)
            learned = [r for r in task["rows"] if r[0].startswith("learned")]
            lbest = min(learned, key=lambda r: r[1])
            note = next((n for k, n in HOLDOUT_NOTE.items() if k in task["name"]), "")
            detail = "".join(
                f'<tr class="d"><td>{html.escape(n)}</td>'
                f'<td class="num">{fmt(l)}</td><td class="num">{a:.3f}</td>'
                f'<td colspan="2"></td></tr>'
                for n, l, a in task["rows"])
            rows_html += f"""
<tbody>
<tr class="t">
  <td><span class="task">{html.escape(task["name"])}</span>
      {f'<span class="note">{note}</span>' if note else ''}</td>
  <td class="num strong">{fmt(lbest[1])}</td>
  <td class="num">{fmt(bb[1])} <span class="who">{html.escape(bb[0].split(" (")[0])}</span></td>
  <td class="bars"><div class="bar l" style="width:{bar(lbest[1]):.0f}%"></div>
      <div class="bar b" style="width:{bar(bb[1]):.0f}%"></div></td>
  <td><span class="chip {v}">{v}</span></td>
</tr>
{detail}
</tbody>"""
        sections += f"""
<section>
  <h2>{b}-step budget <span class="tally">{chips(tallies[b])}</span></h2>
  <div class="scroll"><table>
    <thead><tr><th>task</th><th>learned (best of zs/&lambda;)</th>
    <th>best tuned baseline</th><th>lower&nbsp;loss&nbsp;&rarr;&nbsp;longer&nbsp;bar</th><th></th></tr></thead>
    {rows_html}
  </table></div>
</section>"""

    miles = "".join(f'<li><span class="v">{v}</span>{html.escape(d)}</li>'
                    for v, d in MILESTONES)

    page = f"""<title>vectornet scoreboard</title>
<style>
:root {{
  --bg:#FAFBF9; --panel:#F1F4F1; --ink:#182126; --mut:#5B6B6E; --line:#D8DFDC;
  --acc:#0E7C86; --win:#2F7D46; --tie:#8A6D1F; --loss:#B0453A;
  --win-bg:#E3F0E6; --tie-bg:#F4EDD8; --loss-bg:#F6E4E1; --barb:#B9C6C4;
}}
@media (prefers-color-scheme: dark) {{ :root {{
  --bg:#10151A; --panel:#171E24; --ink:#E6EAEC; --mut:#8FA0A4; --line:#28323A;
  --acc:#53CFDA; --win:#5DBE7E; --tie:#D4B24A; --loss:#E07B6E;
  --win-bg:#1A2B20; --tie-bg:#2A2416; --loss-bg:#2E1D1A; --barb:#3A4850;
}} }}
:root[data-theme="dark"] {{
  --bg:#10151A; --panel:#171E24; --ink:#E6EAEC; --mut:#8FA0A4; --line:#28323A;
  --acc:#53CFDA; --win:#5DBE7E; --tie:#D4B24A; --loss:#E07B6E;
  --win-bg:#1A2B20; --tie-bg:#2A2416; --loss-bg:#2E1D1A; --barb:#3A4850;
}}
:root[data-theme="light"] {{
  --bg:#FAFBF9; --panel:#F1F4F1; --ink:#182126; --mut:#5B6B6E; --line:#D8DFDC;
  --acc:#0E7C86; --win:#2F7D46; --tie:#8A6D1F; --loss:#B0453A;
  --win-bg:#E3F0E6; --tie-bg:#F4EDD8; --loss-bg:#F6E4E1; --barb:#B9C6C4;
}}
* {{ box-sizing:border-box }}
body {{ background:var(--bg); color:var(--ink); margin:0;
  font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1080px; margin:0 auto; padding:40px 28px 80px; }}
.eyebrow {{ font-size:11px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--acc); font-weight:600; }}
h1 {{ font-size:30px; margin:6px 0 4px; letter-spacing:-.01em; text-wrap:balance; }}
.sub {{ color:var(--mut); margin:0 0 10px; }}
.idbar {{ display:flex; flex-wrap:wrap; gap:10px 26px; align-items:baseline;
  border-top:1px solid var(--line); border-bottom:1px solid var(--line);
  padding:12px 0; margin:20px 0 8px; font-family:ui-monospace,Menlo,Consolas,monospace;
  font-size:13px; color:var(--mut); }}
.idbar b {{ color:var(--ink); font-weight:600; }}
h2 {{ font-size:18px; margin:38px 0 12px; display:flex; align-items:center; gap:14px;
  flex-wrap:wrap; }}
.tally {{ display:inline-flex; gap:6px; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; min-width:760px; }}
th {{ text-align:left; font-size:11px; letter-spacing:.08em; text-transform:uppercase;
  color:var(--mut); font-weight:600; padding:6px 14px 6px 0;
  border-bottom:1px solid var(--line); }}
td {{ padding:7px 14px 7px 0; border-bottom:1px solid var(--line);
  vertical-align:middle; }}
tr.d td {{ border-bottom:none; padding:1px 14px 1px 0; color:var(--mut);
  font-size:12.5px; }}
tr.d td:first-child {{ padding-left:18px; }}
tbody tr.d:last-child td {{ padding-bottom:10px; }}
.task {{ font-weight:600; }}
.note {{ font-size:11px; color:var(--acc); border:1px solid var(--acc);
  border-radius:9px; padding:0 7px; margin-left:8px; white-space:nowrap;
  opacity:.85; }}
.num {{ font-family:ui-monospace,Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums; white-space:nowrap; }}
.strong {{ color:var(--acc); font-weight:700; }}
.who {{ color:var(--mut); font-size:12px; }}
.bars {{ min-width:150px; }}
.bar {{ height:5px; border-radius:2px; margin:2px 0; }}
.bar.l {{ background:var(--acc); }}
.bar.b {{ background:var(--barb); }}
.chip {{ font-size:11px; font-weight:700; letter-spacing:.06em;
  text-transform:uppercase; border-radius:10px; padding:2px 9px;
  white-space:nowrap; }}
.chip.win {{ color:var(--win); background:var(--win-bg); }}
.chip.tie {{ color:var(--tie); background:var(--tie-bg); }}
.chip.loss {{ color:var(--loss); background:var(--loss-bg); }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px;
  padding:16px 20px; margin-top:34px; }}
.panel h3 {{ margin:0 0 8px; font-size:14px; }}
.panel p {{ margin:6px 0; color:var(--mut); font-size:13.5px; max-width:72ch; }}
ol.miles {{ margin:8px 0 0; padding:0; list-style:none; }}
ol.miles li {{ padding:5px 0; border-top:1px dashed var(--line); font-size:13.5px;
  color:var(--mut); }}
ol.miles .v {{ font-family:ui-monospace,Menlo,monospace; color:var(--acc);
  font-weight:700; margin-right:12px; }}
footer {{ margin-top:44px; color:var(--mut); font-size:12px;
  border-top:1px solid var(--line); padding-top:12px; }}
</style>
<main>
  <div class="eyebrow">vectornet &middot; learned optimizer research</div>
  <h1>Matrix optimizer vs. tuned baselines</h1>
  <p class="sub">One learned rule, frozen weights, evaluated zero-shot across held-out
  datasets, architectures, and batch sizes &mdash; against per-task lr-tuned Muon,
  Adam, SGD and self-tuning Prodigy. &ldquo;learned&rdquo; shows the better of
  zero-shot and single-scalar &lambda;; sub-rows give every contender.</p>
  <div class="idbar">
    <span>checkpoint <b>{html.escape(checkpoint)}</b></span>
    <span>meta-params <b>28,543</b> (all dimension-free)</span>
    <span>table <b>{html.escape(os.path.basename(eval_path))}</b></span>
    <span>rev <b>{rev}</b></span>
    <span>updated <b>{now}</b></span>
  </div>
  {sections}
  <div class="panel">
    <h3>Scale benchmark (the open front)</h3>
    <p>10.7M-param char-GPT on text8 (d384 &middot; 6 layers &middot; ctx 128,
    stochastic batches), 2000 training steps &mdash; 100&times; beyond meta-training.
    v10 (&mu;P fan-in gauge): <b>zero-shot now descends and holds early-phase parity</b>
    (through ~200 steps), reaching 2.42 (&lambda;=0.5: 2.36) &mdash; but plateaus near
    step 1000, ~2.5&times; its longest meta-training episode, while tuned Muon (1.31)
    and AdamW+cosine (1.30) keep descending. Width and noise are solved; <b>horizon is
    the binding constraint</b>. v11 trains episodes to 2000 steps. Wall-clock:
    215&nbsp;ms/step vs Muon 108 on the RTX&nbsp;3060.</p>
  </div>
  <div class="panel">
    <h3>Milestones</h3>
    <ol class="miles">{miles}</ol>
  </div>
  <footer>Meta-trained on an RTX&nbsp;3060 &middot; PES (evolution strategies), no
  backprop through unrolls &middot; held out everywhere: Fashion-MNIST, pendigits,
  text8, 13 OpenML datasets, MLP shapes h64 &amp; (32,32). Repo:
  murbard/vectornet.</footer>
</main>"""
    with open(out, "w") as fh:
        fh.write(page)
    print(f"wrote {out} from {eval_path} (rev {rev})")
    import shutil
    served = os.path.expanduser("~/vectornet_report/index.html")
    if os.path.isdir(os.path.dirname(served)):  # tailnet-served copy, if set up
        shutil.copy(out, served)
        print(f"refreshed {served}")
    shutil.copy(out, "scoreboard.html")  # artifact-published copy (claude.ai)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="report.html")
    args = ap.parse_args()
    path = args.eval or max(glob.glob("eval_matrix*.txt"), key=os.path.getmtime)
    ck = args.checkpoint or ("learned_matrix" +
                             re.search(r"(\d+)", os.path.basename(path)).group(1) + ".pt")
    render(parse_eval(path), path, ck, args.out)
