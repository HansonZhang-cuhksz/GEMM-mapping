"""Generate the PREFILL throughput-vs-tokens artifact (prefill_throughput.html) from prefill_configs.json."""
import json

d = json.load(open("prefill_configs.json"))
batches = d["batches"]
curves = d["curves"]
BEST = "S3-N4-r2f"; UNFUSED = "S1-N0-r2s"

configs = []
for name, curve in curves.items():
    fam = "f6" if "-N6-" in name else "amort"
    tput = [round((p["tput"] / 1e6) if p["tput"] else 0, 4) for p in curve]
    configs.append({"name": name, "fam": fam, "tput": tput, "best": name == BEST, "unfused": name == UNFUSED})
data = {"batches": batches, "configs": configs}

def peak(name): return max(curves[name], key=lambda p: p["tput"] or 0)
best_p = peak(BEST); best_B = best_p["B"]; best_tp = best_p["tput"] / 1e6
unf_tp = peak(UNFUSED)["tput"] / 1e6
f6_tp = max(peak(n)["tput"] for n in curves if "-N6-" in n) / 1e6
# margin curve peak
margins = [(curves[BEST][i]["tput"] / curves[UNFUSED][i]["tput"]) for i in range(len(batches))
           if curves[BEST][i]["tput"] and curves[UNFUSED][i]["tput"]]
peak_margin = (max(margins) - 1) * 100

HTML = f"""<title>Prefill fusion configuration throughput</title>
<style>
:root {{
  --surface: #fcfcfb; --panel: #ffffff; --ink: #14181d; --muted: #5c6570; --faint: #8b95a1;
  --grid: #e9ebef; --hair: #e0e3e8; --amort: #1f9e8f; --f6: #d1615d; --best: #3a6fd0;
  --amort-soft: rgba(31,158,143,.30); --f6-soft: rgba(209,97,93,.42);
  --shadow: 0 1px 2px rgba(20,24,29,.06), 0 8px 24px rgba(20,24,29,.05);
}}
@media (prefers-color-scheme: dark) {{ :root {{
  --surface: #0f1319; --panel: #151a21; --ink: #e7eaef; --muted: #939eaa; --faint: #6b7684;
  --grid: #222932; --hair: #2a323c; --amort: #35a898; --f6: #d9706c; --best: #5580dd;
  --amort-soft: rgba(53,168,152,.34); --f6-soft: rgba(217,112,108,.44);
  --shadow: 0 1px 2px rgba(0,0,0,.3), 0 10px 30px rgba(0,0,0,.35); }} }}
:root[data-theme="light"] {{
  --surface: #fcfcfb; --panel: #ffffff; --ink: #14181d; --muted: #5c6570; --faint: #8b95a1;
  --grid: #e9ebef; --hair: #e0e3e8; --amort: #1f9e8f; --f6: #d1615d; --best: #3a6fd0;
  --amort-soft: rgba(31,158,143,.30); --f6-soft: rgba(209,97,93,.42);
  --shadow: 0 1px 2px rgba(20,24,29,.06), 0 8px 24px rgba(20,24,29,.05); }}
:root[data-theme="dark"] {{
  --surface: #0f1319; --panel: #151a21; --ink: #e7eaef; --muted: #939eaa; --faint: #6b7684;
  --grid: #222932; --hair: #2a323c; --amort: #35a898; --f6: #d9706c; --best: #5580dd;
  --amort-soft: rgba(53,168,152,.34); --f6-soft: rgba(217,112,108,.44);
  --shadow: 0 1px 2px rgba(0,0,0,.3), 0 10px 30px rgba(0,0,0,.35); }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--surface); color: var(--ink);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased; line-height: 1.55; }}
.wrap {{ max-width: 1000px; margin: 0 auto; padding: 44px 24px 72px; }}
.mono {{ font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace; }}
.eyebrow {{ font-size: 12px; letter-spacing: .13em; text-transform: uppercase; color: var(--faint); font-weight: 600; font-family: ui-monospace, monospace; }}
h1 {{ font-size: clamp(28px, 4vw, 40px); line-height: 1.08; margin: 10px 0 8px; letter-spacing: -.02em; text-wrap: balance; font-weight: 680; }}
.lede {{ font-size: 17px; color: var(--muted); max-width: 64ch; margin: 0; }}
.stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 32px 0 8px; }}
@media (max-width: 680px) {{ .stats {{ grid-template-columns: repeat(2, 1fr); }} }}
.stat {{ background: var(--panel); border: 1px solid var(--hair); border-radius: 12px; padding: 16px 16px 14px; }}
.stat .k {{ font-size: 12px; color: var(--faint); letter-spacing: .04em; text-transform: uppercase; font-weight: 600; }}
.stat .v {{ font-size: 26px; font-weight: 680; margin-top: 6px; font-variant-numeric: tabular-nums; letter-spacing: -.01em; }}
.stat .u {{ font-size: 13px; color: var(--muted); font-weight: 500; }}
.card {{ background: var(--panel); border: 1px solid var(--hair); border-radius: 16px; padding: 22px 22px 10px; margin-top: 22px; box-shadow: var(--shadow); }}
.card h2 {{ font-size: 15px; margin: 0; font-weight: 660; }}
.card .sub {{ font-size: 13px; color: var(--muted); margin: 3px 0 6px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 16px; margin: 4px 0 6px; font-size: 13px; }}
.legend span {{ display: inline-flex; align-items: center; gap: 7px; color: var(--muted); }}
.swatch {{ width: 22px; height: 3px; border-radius: 2px; flex: none; }}
.swatch.dash {{ height: 0; border-top: 2px dashed var(--faint); }}
svg {{ width: 100%; height: auto; display: block; touch-action: none; }}
.tick {{ font-size: 11px; fill: var(--faint); font-family: ui-monospace, monospace; }}
.axtitle {{ font-size: 12px; fill: var(--muted); font-weight: 600; }}
.tip {{ position: absolute; pointer-events: none; background: var(--panel); border: 1px solid var(--hair); border-radius: 10px; padding: 9px 11px; font-size: 12.5px; box-shadow: var(--shadow); opacity: 0; transition: opacity .08s; min-width: 178px; }}
.tip .b {{ font-weight: 680; font-variant-numeric: tabular-nums; }}
.tip .row {{ display: flex; justify-content: space-between; gap: 14px; margin-top: 3px; color: var(--muted); }}
.tip .row b {{ color: var(--ink); font-variant-numeric: tabular-nums; }}
.note {{ display: grid; grid-template-columns: 1.3fr 1fr; gap: 18px; margin-top: 22px; }}
@media (max-width: 680px) {{ .note {{ grid-template-columns: 1fr; }} }}
.take {{ background: var(--panel); border: 1px solid var(--hair); border-radius: 16px; padding: 20px 22px; }}
.take h3 {{ margin: 0 0 8px; font-size: 14px; }}
.take p {{ margin: 0 0 10px; font-size: 14.5px; color: var(--ink); }}
.take .win {{ color: var(--best); font-weight: 640; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--grid); font-variant-numeric: tabular-nums; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: var(--faint); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }}
td.name {{ font-family: ui-monospace, monospace; }}
.dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }}
footer {{ margin-top: 30px; font-size: 12.5px; color: var(--faint); }}
</style>

<div class="wrap">
  <div class="eyebrow">GLM-5.2 MoE · prefill (incl. O(T²) attention) · H100 · snowcat-roofline estimation</div>
  <h1>Prefill: every fusion configuration, ranked by throughput</h1>
  <p class="lede">The same 40 configurations as decode, but the layer now carries the full causal
  attention core. Throughput rises with sequence length — then <b>falls</b> as O(T²) attention takes
  over, and the fusion advantage fades with it. Tokens per second; higher is better.</p>

  <div class="stats">
    <div class="stat"><div class="k">Best config</div><div class="v mono">{BEST}</div><div class="u">same as decode</div></div>
    <div class="stat"><div class="k">Peak throughput</div><div class="v">{best_tp:.3f}<span class="u"> Mtok/s</span></div><div class="u">at {best_B:,} tokens</div></div>
    <div class="stat"><div class="k">Fusion gain (peak)</div><div class="v">+{peak_margin:.0f}<span class="u">%</span></div><div class="u">vs unfused; fades at long T</div></div>
    <div class="stat"><div class="k">vs best on-chip FFN</div><div class="v">{best_tp/f6_tp:.1f}×</div><div class="u">F6 worst family</div></div>
  </div>

  <div class="card">
    <h2>Throughput vs sequence length (prefill tokens)</h2>
    <div class="sub">Each line is one configuration. Throughput peaks near {best_B:,} tokens, then declines as the O(T²) attention core dominates. Two families: amortized GEMMs vs on-chip F6.</div>
    <div class="legend">
      <span><span class="swatch" style="background:var(--best)"></span> Best — {BEST}</span>
      <span><span class="swatch" style="background:var(--amort)"></span> Amortized GEMMs (30)</span>
      <span><span class="swatch" style="background:var(--f6)"></span> On-chip F6 (10)</span>
      <span><span class="swatch dash"></span> Fully unfused</span>
    </div>
    <div style="position:relative">
      <svg id="chart" viewBox="0 0 900 430" role="img" aria-label="Prefill throughput versus sequence length for all fusion configurations"></svg>
      <div class="tip" id="tip"></div>
    </div>
  </div>

  <div class="note">
    <div class="take">
      <h3>Prefill vs decode</h3>
      <p>The winner is the <span class="win mono">same config as decode</span> — fold all four vector
      ops into GEMM epilogues, keep up_gate and down as weight-amortized grouped GEMMs. But fusion is
      worth <b>less</b> in prefill: a peak of <b>+{peak_margin:.0f}%</b> near {best_B//1024}k tokens,
      shrinking toward ~+1% at 128k — versus <b>+6%</b> in decode.</p>
      <p style="color:var(--muted);font-size:13.5px">Two reasons: (1) the <b>O(T²) attention</b> grows
      from ~0% of the layer at 512 tokens to <b>~78%</b> at 128k, diluting the FFN the fusion targets;
      (2) at prefill scale the FFN GEMMs are <b>compute-bound</b> (tokens/expert = T/32 is large), so the
      eliminated vector kernels hide under the GEMM math. On-chip F6 is still the worst (~3× off).</p>
    </div>
    <div class="take">
      <h3>Top &amp; bottom</h3>
      <table id="rank"><thead><tr><th>config</th><th>peak Mtok/s</th><th>@ tokens</th></tr></thead><tbody></tbody></table>
    </div>
  </div>

  <footer>Analytical snowcat-roofline estimate. Attention core modeled as causal flash (N_HEADS·T²·(qk 192 + v 256),
  compute-bound). Layer = attention prefix + mla_o → router → FFN → residual₂; the prefix is uniform across all 40
  configs, so it shifts absolute throughput but not the ranking. Peak token count is tile-quantization-sensitive.</footer>
</div>

<script>
const DATA = {json.dumps(data)};
const BEST = "{BEST}";
const svg = document.getElementById('chart'), tip = document.getElementById('tip');
const W = 900, H = 430, ml = 58, mr = 84, mt = 18, mb = 46;
const bx = DATA.batches, x0 = Math.log2(bx[0]), x1 = Math.log2(bx[bx.length-1]);
const ymax = {max(0.78, best_tp*1.06):.3f};
const X = b => ml + (Math.log2(b) - x0) / (x1 - x0) * (W - ml - mr);
const Y = v => mt + (1 - v / ymax) * (H - mt - mb);
const NS = 'http://www.w3.org/2000/svg';
const el = (t, a) => {{ const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); return e; }};
for (let v = 0; v <= ymax-0.05; v += 0.2) {{
  svg.appendChild(el('line', {{x1: ml, x2: W-mr, y1: Y(v), y2: Y(v), stroke: 'var(--grid)', 'stroke-width': 1}}));
  const t = el('text', {{x: ml-8, y: Y(v)+3.5, class: 'tick', 'text-anchor': 'end'}}); t.textContent = v.toFixed(1); svg.appendChild(t);
}}
[512,2048,8192,32768,131072].forEach(b => {{
  const t = el('text', {{x: X(b), y: H-mb+18, class: 'tick', 'text-anchor': 'middle'}}); t.textContent = b>=1024?(b/1024+'k'):b; svg.appendChild(t);
}});
let ax = el('text', {{x: ml, y: H-6, class: 'axtitle'}}); ax.textContent = 'sequence length (prefill tokens, log)'; svg.appendChild(ax);
let ay = el('text', {{class: 'axtitle', transform: `translate(15,${{mt+8}}) rotate(-90)`, 'text-anchor': 'end'}}); ay.textContent = 'throughput  (Mtok/s)'; svg.appendChild(ay);

const path = c => 'M' + c.tput.map((v,i) => X(bx[i]) + ',' + Y(v)).join(' L');
const draw = (filter, attrs) => DATA.configs.filter(filter).forEach(c => svg.appendChild(el('path', {{d: path(c), fill: 'none', ...attrs}})));
draw(c => c.fam==='amort' && !c.best, {{stroke: 'var(--amort-soft)', 'stroke-width': 1.2}});
draw(c => c.fam==='f6', {{stroke: 'var(--f6-soft)', 'stroke-width': 1.2}});
DATA.configs.filter(c => c.unfused).forEach(c => svg.appendChild(el('path', {{d: path(c), fill: 'none', stroke: 'var(--faint)', 'stroke-width': 1.6, 'stroke-dasharray': '5 4'}})));
const best = DATA.configs.find(c => c.best);
svg.appendChild(el('path', {{d: path(best), fill: 'none', stroke: 'var(--best)', 'stroke-width': 2.8, 'stroke-linejoin': 'round'}}));
const pi = best.tput.indexOf(Math.max(...best.tput));
svg.appendChild(el('circle', {{cx: X(bx[pi]), cy: Y(best.tput[pi]), r: 4.5, fill: 'var(--best)', stroke: 'var(--panel)', 'stroke-width': 2}}));
let lb = el('text', {{x: X(bx[pi]), y: Y(best.tput[pi])-10, class: 'tick', fill: 'var(--best)', 'font-weight': 700, 'text-anchor': 'middle'}});
lb.textContent = 'peak ' + best.tput[pi].toFixed(3); svg.appendChild(lb);

const cross = el('line', {{y1: mt, y2: H-mb, stroke: 'var(--ink)', 'stroke-width': 1, opacity: 0}}); svg.appendChild(cross);
svg.addEventListener('pointermove', ev => {{
  const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) / r.width * W;
  let bi = 0, bd = 1e9; bx.forEach((b,i) => {{ const dd = Math.abs(X(b)-px); if (dd<bd) {{ bd=dd; bi=i; }} }});
  const B = bx[bi];
  cross.setAttribute('x1', X(B)); cross.setAttribute('x2', X(B)); cross.setAttribute('opacity', .5);
  const am = DATA.configs.filter(c=>c.fam==='amort').map(c=>c.tput[bi]);
  const f6 = DATA.configs.filter(c=>c.fam==='f6').map(c=>c.tput[bi]).filter(v=>v>0);
  const uf = DATA.configs.find(c=>c.unfused).tput[bi];
  tip.innerHTML = `<div class="b">${{B.toLocaleString()}} tokens · ${{(B*8/256)|0}} tok/expert</div>`+
    `<div class="row"><span><span class="dot" style="background:var(--best)"></span>best</span><b>${{best.tput[bi].toFixed(3)}}</b></div>`+
    `<div class="row"><span>vs unfused</span><b>+${{((best.tput[bi]/uf-1)*100).toFixed(1)}}%</b></div>`+
    `<div class="row"><span><span class="dot" style="background:var(--f6)"></span>on-chip F6</span><b>${{f6.length?Math.max(...f6).toFixed(3):'—'}}</b></div>`;
  const left = Math.min(X(B)/W*r.width+14, r.width-186); tip.style.left = left + 'px'; tip.style.top = '6px'; tip.style.opacity = 1;
}});
svg.addEventListener('pointerleave', () => {{ tip.style.opacity = 0; cross.setAttribute('opacity', 0); }});

const ranked = DATA.configs.map(c => ({{name: c.name, fam: c.fam, peak: Math.max(...c.tput), B: bx[c.tput.indexOf(Math.max(...c.tput))]}})).sort((a,b)=>b.peak-a.peak);
const tb = document.querySelector('#rank tbody');
[...ranked.slice(0,5), null, ...ranked.slice(-2)].forEach(r => {{
  const tr = document.createElement('tr');
  if (!r) {{ tr.innerHTML = `<td colspan="3" style="text-align:center;color:var(--faint)">⋯ 33 more ⋯</td>`; tb.appendChild(tr); return; }}
  const col = r.fam==='f6'?'var(--f6)':(r.name===BEST?'var(--best)':'var(--amort)');
  tr.innerHTML = `<td class="name"><span class="dot" style="background:${{col}}"></span>${{r.name}}</td><td>${{r.peak.toFixed(3)}}</td><td>${{r.B.toLocaleString()}}</td>`;
  tb.appendChild(tr);
}});
</script>
"""
open("prefill_throughput.html", "w").write(HTML)
print("wrote prefill_throughput.html", len(HTML), "bytes")
