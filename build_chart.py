"""Generate the throughput-vs-batch artifact (fusion_throughput.html) from fusion_configs.json."""
import json

d = json.load(open("fusion_configs.json"))
batches = d["batches"]
curves = d["curves"]
BEST = "S3-N4-r2f"
UNFUSED = "S1-N0-r2s"

configs = []
for name, curve in curves.items():
    fam = "f6" if "-N6-" in name else "amort"
    tput = [round((p["tput"] / 1e6) if p["tput"] else 0, 4) for p in curve]
    configs.append({"name": name, "fam": fam, "tput": tput,
                    "best": name == BEST, "unfused": name == UNFUSED})
# order: amort first, f6 last, best drawn on top (handled in JS)
data = {"batches": batches, "configs": configs}

def cfg_desc(name):
    a, f, r = name.split("-")
    attn = {"S1": "residual + RMSNorm separate", "S2": "residual→mla_o epilogue",
            "S3": "residual + RMSNorm→mla_o epilogue", "S4": "RMSNorm→up_gate prologue",
            "S5": "residual→mla_o, RMSNorm→up_gate prologue"}[a]
    ffn = {"N0": "up_gate + SwiGLU + down all separate", "N4": "SwiGLU→up_gate epilogue (½-width)",
           "N5": "SwiGLU→down prologue (2×-wide)", "N6": "on-chip full FFN (weights re-read)"}[f]
    res2 = "residual₂ fused" if r == "r2f" else "residual₂ separate"
    return attn, ffn, res2

best_attn, best_ffn, best_res2 = cfg_desc(BEST)

# peak stats
def peak(name): return max(curves[name], key=lambda p: p["tput"] or 0)
best_peak = peak(BEST); best_B = best_peak["B"]; best_tp = best_peak["tput"] / 1e6
unf_peak = peak(UNFUSED)["tput"] / 1e6
f6_peak = max(peak(n)["tput"] for n in curves if "-N6-" in n) / 1e6

HTML = f"""<style>
:root {{
  --surface: #fcfcfb; --panel: #ffffff; --ink: #14181d; --muted: #5c6570; --faint: #8b95a1;
  --grid: #e9ebef; --hair: #e0e3e8; --amort: #1f9e8f; --f6: #d1615d; --best: #3a6fd0;
  --amort-soft: rgba(31,158,143,.30); --f6-soft: rgba(209,97,93,.42);
  --shadow: 0 1px 2px rgba(20,24,29,.06), 0 8px 24px rgba(20,24,29,.05);
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --surface: #0f1319; --panel: #151a21; --ink: #e7eaef; --muted: #939eaa;
    --faint: #6b7684; --grid: #222932; --hair: #2a323c; --amort: #35a898; --f6: #d9706c; --best: #5580dd;
    --amort-soft: rgba(53,168,152,.34); --f6-soft: rgba(217,112,108,.44);
    --shadow: 0 1px 2px rgba(0,0,0,.3), 0 10px 30px rgba(0,0,0,.35);
  }}
}}
:root[data-theme="light"] {{
  --surface: #fcfcfb; --panel: #ffffff; --ink: #14181d; --muted: #5c6570; --faint: #8b95a1;
  --grid: #e9ebef; --hair: #e0e3e8; --amort: #1f9e8f; --f6: #d1615d; --best: #3a6fd0;
  --amort-soft: rgba(31,158,143,.30); --f6-soft: rgba(209,97,93,.42);
  --shadow: 0 1px 2px rgba(20,24,29,.06), 0 8px 24px rgba(20,24,29,.05);
}}
:root[data-theme="dark"] {{
  --surface: #0f1319; --panel: #151a21; --ink: #e7eaef; --muted: #939eaa; --faint: #6b7684;
  --grid: #222932; --hair: #2a323c; --amort: #35a898; --f6: #d9706c; --best: #5580dd;
  --amort-soft: rgba(53,168,152,.34); --f6-soft: rgba(217,112,108,.44);
  --shadow: 0 1px 2px rgba(0,0,0,.3), 0 10px 30px rgba(0,0,0,.35);
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--surface); color: var(--ink);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased; line-height: 1.55; }}
.wrap {{ max-width: 1000px; margin: 0 auto; padding: 44px 24px 72px; }}
.mono {{ font-family: ui-monospace, "SF Mono", "JetBrains Mono", "Cascadia Code", Menlo, monospace; }}
.eyebrow {{ font-size: 12px; letter-spacing: .13em; text-transform: uppercase; color: var(--faint);
  font-weight: 600; font-family: ui-monospace, monospace; }}
h1 {{ font-size: clamp(28px, 4vw, 40px); line-height: 1.08; margin: 10px 0 8px; letter-spacing: -.02em;
  text-wrap: balance; font-weight: 680; }}
.lede {{ font-size: 17px; color: var(--muted); max-width: 62ch; margin: 0; }}
.stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 32px 0 8px; }}
@media (max-width: 680px) {{ .stats {{ grid-template-columns: repeat(2, 1fr); }} }}
.stat {{ background: var(--panel); border: 1px solid var(--hair); border-radius: 12px; padding: 16px 16px 14px; }}
.stat .k {{ font-size: 12px; color: var(--faint); letter-spacing: .04em; text-transform: uppercase; font-weight: 600; }}
.stat .v {{ font-size: 26px; font-weight: 680; margin-top: 6px; font-variant-numeric: tabular-nums; letter-spacing: -.01em; }}
.stat .u {{ font-size: 13px; color: var(--muted); font-weight: 500; }}
.card {{ background: var(--panel); border: 1px solid var(--hair); border-radius: 16px; padding: 22px 22px 10px;
  margin-top: 22px; box-shadow: var(--shadow); }}
.card h2 {{ font-size: 15px; margin: 0; font-weight: 660; letter-spacing: -.005em; }}
.card .sub {{ font-size: 13px; color: var(--muted); margin: 3px 0 6px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 16px; margin: 4px 0 6px; font-size: 13px; }}
.legend span {{ display: inline-flex; align-items: center; gap: 7px; color: var(--muted); }}
.swatch {{ width: 22px; height: 3px; border-radius: 2px; flex: none; }}
.swatch.dash {{ height: 0; border-top: 2px dashed var(--faint); }}
svg {{ width: 100%; height: auto; display: block; touch-action: none; }}
.tick {{ font-size: 11px; fill: var(--faint); font-family: ui-monospace, monospace; }}
.axtitle {{ font-size: 12px; fill: var(--muted); font-weight: 600; }}
.tip {{ position: absolute; pointer-events: none; background: var(--panel); border: 1px solid var(--hair);
  border-radius: 10px; padding: 9px 11px; font-size: 12.5px; box-shadow: var(--shadow); opacity: 0;
  transition: opacity .08s; min-width: 168px; }}
.tip .b {{ font-weight: 680; font-variant-numeric: tabular-nums; }}
.tip .row {{ display: flex; justify-content: space-between; gap: 14px; margin-top: 3px; color: var(--muted); }}
.tip .row b {{ color: var(--ink); font-variant-numeric: tabular-nums; }}
.note {{ display: grid; grid-template-columns: 1.3fr 1fr; gap: 18px; margin-top: 22px; }}
@media (max-width: 680px) {{ .note {{ grid-template-columns: 1fr; }} }}
.take {{ background: var(--panel); border: 1px solid var(--hair); border-radius: 16px; padding: 20px 22px; }}
.take h3 {{ margin: 0 0 8px; font-size: 14px; letter-spacing: .02em; }}
.take p {{ margin: 0 0 10px; font-size: 14.5px; color: var(--ink); }}
.take .win {{ color: var(--best); font-weight: 640; }}
.chips {{ display: flex; flex-direction: column; gap: 7px; margin-top: 10px; }}
.chip {{ display: flex; gap: 9px; font-size: 13px; align-items: baseline; }}
.chip .op {{ color: var(--faint); font-family: ui-monospace, monospace; font-size: 11.5px; min-width: 62px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--grid); font-variant-numeric: tabular-nums; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: var(--faint); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }}
td.name {{ font-family: ui-monospace, monospace; }}
.dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }}
footer {{ margin-top: 30px; font-size: 12.5px; color: var(--faint); }}
</style>

<div class="wrap">
  <div class="eyebrow">GLM-5.2 MoE · decode layer · H100 · snowcat-roofline estimation</div>
  <h1>Every fusion configuration, ranked by throughput</h1>
  <p class="lede">All {len(configs)} ways to place the four vector ops (residual, RMSNorm, SwiGLU, residual₂)
  into GEMM epilogues — crossed with keeping the FFN GEMMs weight-amortized or fusing them on-chip —
  swept across batch size. Throughput is tokens per second; higher is better.</p>

  <div class="stats">
    <div class="stat"><div class="k">Best config</div><div class="v mono">{BEST}</div><div class="u">of {len(configs)} enumerated</div></div>
    <div class="stat"><div class="k">Best throughput</div><div class="v">{best_tp:.3f}<span class="u"> Mtok/s</span></div><div class="u">at batch {best_B:,}</div></div>
    <div class="stat"><div class="k">vs fully unfused</div><div class="v">+{(best_tp/unf_peak-1)*100:.0f}<span class="u">%</span></div><div class="u">{unf_peak:.3f} → {best_tp:.3f}</div></div>
    <div class="stat"><div class="k">vs best on-chip FFN</div><div class="v">{best_tp/f6_peak:.1f}×</div><div class="u">F6 capped at {f6_peak:.3f}</div></div>
  </div>

  <div class="card">
    <h2>Throughput vs batch size</h2>
    <div class="sub">Each line is one configuration. Two families separate cleanly: amortized (scales with batch) and on-chip F6 (capped). Batch bounded ≤ 16,384.</div>
    <div class="legend">
      <span><span class="swatch" style="background:var(--best)"></span> Best — {BEST}</span>
      <span><span class="swatch" style="background:var(--amort)"></span> Amortized GEMMs (30 configs)</span>
      <span><span class="swatch" style="background:var(--f6)"></span> On-chip full-FFN / F6 (10 configs)</span>
      <span><span class="swatch dash"></span> Fully unfused baseline</span>
    </div>
    <div style="position:relative">
      <svg id="chart" viewBox="0 0 900 430" role="img" aria-label="Throughput versus batch size for all fusion configurations"></svg>
      <div class="tip" id="tip"></div>
    </div>
  </div>

  <div class="note">
    <div class="take">
      <h3>What the winner does</h3>
      <p>The best config <span class="win mono">{BEST}</span> folds <b>all four vector ops into GEMM
      epilogues</b> while keeping <b>up_gate and down as separate, weight-amortized grouped GEMMs</b>.
      It never re-reads weights, so its throughput scales with batch to <b>{best_tp:.3f} Mtok/s</b>.</p>
      <div class="chips">
        <div class="chip"><span class="op">attn</span><span>{best_attn}</span></div>
        <div class="chip"><span class="op">ffn</span><span>{best_ffn}</span></div>
        <div class="chip"><span class="op">post</span><span>{best_res2}; router stays standalone</span></div>
      </div>
      <p style="margin-top:14px;color:var(--muted);font-size:13.5px">The rule: <b>fuse the cheap vector ops into
      weight-amortized GEMMs; do <i>not</i> make the whole FFN intermediate-resident</b> — that (F6) forfeits
      weight amortization and caps throughput ~{best_tp/f6_peak:.1f}× lower.</p>
    </div>
    <div class="take">
      <h3>Top &amp; bottom</h3>
      <table id="rank"><thead><tr><th>config</th><th>peak Mtok/s</th><th>@ batch</th></tr></thead><tbody></tbody></table>
    </div>
  </div>

  <footer>Analytical snowcat-roofline estimate (not measured). Top cluster (S3/S5/S2 × N4/N5, all r2f) sits within ~1%;
  the meaningful split is amortized-scaling vs on-chip-capped. tokens/expert = batch × top_k(8) / experts(256).</footer>
</div>

<script>
const DATA = {json.dumps(data)};
const BEST = "{BEST}";
const svg = document.getElementById('chart'), tip = document.getElementById('tip');
const W = 900, H = 430, ml = 58, mr = 96, mt = 18, mb = 46;
const bx = DATA.batches, x0 = Math.log2(bx[0]), x1 = Math.log2(bx[bx.length-1]);
const ymax = 1.28;
const X = b => ml + (Math.log2(b) - x0) / (x1 - x0) * (W - ml - mr);
const Y = v => mt + (1 - v / ymax) * (H - mt - mb);
const NS = 'http://www.w3.org/2000/svg';
const el = (t, a) => {{ const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); return e; }};

// grid + y ticks
for (let v = 0; v <= 1.25; v += 0.25) {{
  svg.appendChild(el('line', {{x1: ml, x2: W-mr, y1: Y(v), y2: Y(v), stroke: 'var(--grid)', 'stroke-width': 1}}));
  const t = el('text', {{x: ml-8, y: Y(v)+3.5, class: 'tick', 'text-anchor': 'end'}}); t.textContent = v.toFixed(2); svg.appendChild(t);
}}
// x ticks
[128,512,2048,8192,16384].forEach(b => {{
  const t = el('text', {{x: X(b), y: H-mb+18, class: 'tick', 'text-anchor': 'middle'}}); t.textContent = b>=1024?(b/1024+'k'):b; svg.appendChild(t);
}});
let ax = el('text', {{x: ml, y: H-6, class: 'axtitle'}}); ax.textContent = 'batch size (tokens, log scale)'; svg.appendChild(ax);
let ay = el('text', {{class: 'axtitle', transform: `translate(15,${{mt+8}}) rotate(-90)`, 'text-anchor': 'end'}});
ay.textContent = 'throughput  (Mtok/s)'; svg.appendChild(ay);

const path = c => 'M' + c.tput.map((v,i) => X(bx[i]) + ',' + Y(v)).join(' L');
// draw order: amort family, f6 family, unfused, best on top
const draw = (filter, attrs) => DATA.configs.filter(filter).forEach(c => svg.appendChild(el('path', {{d: path(c), fill: 'none', ...attrs}})));
draw(c => c.fam==='amort' && !c.best, {{stroke: 'var(--amort-soft)', 'stroke-width': 1.2}});
draw(c => c.fam==='f6', {{stroke: 'var(--f6-soft)', 'stroke-width': 1.2}});
DATA.configs.filter(c => c.unfused).forEach(c => svg.appendChild(el('path', {{d: path(c), fill: 'none', stroke: 'var(--faint)', 'stroke-width': 1.6, 'stroke-dasharray': '5 4'}})));
const best = DATA.configs.find(c => c.best);
svg.appendChild(el('path', {{d: path(best), fill: 'none', stroke: 'var(--best)', 'stroke-width': 2.8, 'stroke-linejoin': 'round'}}));
// endpoint marker + label for best
const li = best.tput.length-1, bxi = bx[li], byi = best.tput[li];
svg.appendChild(el('circle', {{cx: X(bxi), cy: Y(byi), r: 4, fill: 'var(--best)', stroke: 'var(--panel)', 'stroke-width': 2}}));
let lb = el('text', {{x: X(bxi)+9, y: Y(best.tput[best.tput.indexOf(Math.max(...best.tput))])-2, class: 'tick', fill: 'var(--best)', 'font-weight': 700}});
lb.textContent = 'best'; svg.appendChild(lb);

// hover crosshair + summary tooltip
const cross = el('line', {{y1: mt, y2: H-mb, stroke: 'var(--ink)', 'stroke-width': 1, opacity: 0}}); svg.appendChild(cross);
const wrapEl = svg.parentElement;
svg.addEventListener('pointermove', ev => {{
  const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) / r.width * W;
  let bi = 0, bd = 1e9;
  bx.forEach((b,i) => {{ const d = Math.abs(X(b)-px); if (d<bd) {{ bd=d; bi=i; }} }});
  const B = bx[bi];
  cross.setAttribute('x1', X(B)); cross.setAttribute('x2', X(B)); cross.setAttribute('opacity', .5);
  const am = DATA.configs.filter(c=>c.fam==='amort').map(c=>c.tput[bi]);
  const f6 = DATA.configs.filter(c=>c.fam==='f6').map(c=>c.tput[bi]).filter(v=>v>0);
  const bestv = best.tput[bi];
  tip.innerHTML = `<div class="b">batch ${{B.toLocaleString()}} · ${{(B*8/256)|0}} tok/expert</div>`+
    `<div class="row"><span><span class="dot" style="background:var(--best)"></span>best</span><b>${{bestv.toFixed(3)}}</b></div>`+
    `<div class="row"><span><span class="dot" style="background:var(--amort)"></span>amortized</span><b>${{Math.min(...am).toFixed(2)}}–${{Math.max(...am).toFixed(2)}}</b></div>`+
    `<div class="row"><span><span class="dot" style="background:var(--f6)"></span>on-chip F6</span><b>${{f6.length?Math.max(...f6).toFixed(3):'—'}}</b></div>`;
  const tx = Math.min(X(B)+14, W-150), left = tx/W*r.width;
  tip.style.left = Math.min(left, r.width-176) + 'px'; tip.style.top = (Y(ymax)*r.height/H + 4) + 'px'; tip.style.opacity = 1;
}});
svg.addEventListener('pointerleave', () => {{ tip.style.opacity = 0; cross.setAttribute('opacity', 0); }});

// ranking table: top 5 + bottom 3
const ranked = DATA.configs.map(c => ({{name: c.name, fam: c.fam, peak: Math.max(...c.tput), B: bx[c.tput.indexOf(Math.max(...c.tput))]}})).sort((a,b)=>b.peak-a.peak);
const tb = document.querySelector('#rank tbody');
const rows = [...ranked.slice(0,5), null, ...ranked.slice(-2)];
rows.forEach(r => {{
  const tr = document.createElement('tr');
  if (!r) {{ tr.innerHTML = `<td colspan="3" style="text-align:center;color:var(--faint)">⋯ 33 more ⋯</td>`; tb.appendChild(tr); return; }}
  const col = r.fam==='f6'?'var(--f6)':(r.name===BEST?'var(--best)':'var(--amort)');
  tr.innerHTML = `<td class="name"><span class="dot" style="background:${{col}}"></span>${{r.name}}</td><td>${{r.peak.toFixed(3)}}</td><td>${{r.B.toLocaleString()}}</td>`;
  tb.appendChild(tr);
}});
</script>
"""

open("fusion_throughput.html", "w").write(HTML)
print("wrote fusion_throughput.html", len(HTML), "bytes")
