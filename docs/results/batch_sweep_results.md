# Batch size × fusion: unfused vs fused throughput (GLM-5.2 MoE decode, H100, estimation)

Sweep decode batch B; throughput = tokens/s = B / layer_time. Full decode layer, MoE (256 experts,
top_k 8 → tokens/expert = B/32). Estimation via the snowcat-roofline model, reusing
`fusion_time_estimator`'s validated F2/F4/F6 physics (`batch_sweep.py`). Verified by an adversarial
workflow that corrected the headline. **Estimation, not measured.**

## Result (H100)

| batch | tok/exp | unfused Mtok/s | **F6** (on-chip FFN) | **smart** (epilogue-only) |
|---:|---:|---:|---:|---:|
| 512 | 16 | 0.086 | 0.086 (1.01×) | 0.086 (1.00×) |
| 1024 | 32 | 0.166 | 0.168 (1.01×) | 0.167 (1.01×) |
| 2048 | 64 | 0.312 | 0.168 (**0.54×**) | 0.317 (1.02×) |
| 8192 | 256 | 0.919 | 0.168 (0.18×) | 0.963 (1.05×) |
| ≥16384 | ≥512 | **1.104** (plateau) | 0.168 (flat) | **1.155** (1.05×) |

**Optimal batch & best throughput:**
- **Unfused → large batch: best 1.104 Mtok/s** (plateau begins B≥16384, when the FFN GEMMs turn compute-bound).
- **F6 on-chip fusion → small batch: best ~0.17–0.25 Mtok/s** (0.168 on the power-of-2 grid; 0.253 at the refined optimum B=1568, Me=49, mt=1). **4.4–6.6× below unfused's best.**
- **Smart epilogue-only fusion → large batch: best 1.155 Mtok/s (1.047× > unfused).**

## Your hypothesis: confirmed for the *aggressive* fusion, with an important caveat

**Directionally correct:** the unfused pass's optimum is large batch, and the **aggressive on-chip
full-FFN fusion (F6) is only competitive at small batch** (wins head-to-head only for B ≤ 1568, a
sharp divisor-driven cliff at the mt 1→2 jump). So "unfused prefers large batch, fused prefers small
batch" holds exactly — **for F6.**

**But the strong corollary "fusion is throughput-capped / ~6.6× worse" is a strawman.** It is true
*only* of F6, and for one structural reason: **F6 forfeits weight amortization.** Holding the whole
FFN intermediate on-chip forces it to re-read both weight matrices `mt = Me/m0 ∝ B` times, so its time
grows ∝ B → **throughput capped, cannot scale.** The unfused pass reads each expert weight *once* and
amortizes it over all B/32 routed tokens → throughput *scales* with batch until the compute roofline.

**A cheaper fusion has no such tradeoff.** The "smart" pass fuses only the vector ops (F2 attention +
SwiGLU into the up_gate epilogue) while keeping up_gate and down as **weight-amortized grouped GEMMs**.
It inherits unfused's amortization *and* saves the residual/RMSNorm/SwiGLU HBM round-trips, so it
**scales with batch and slightly beats unfused at every batch** (1.155 vs 1.104 Mtok/s, 1.047×) — and,
like unfused, it prefers *large* batch. So fusion as a category is **not** throughput-capped or
small-batch-preferring; only the intermediate-resident F6 is.

## Is there any regime where F6's best beats unfused's best?

No (of 9 tested). Levers shrink the gap but never close it — narrower INTERMEDIATE (INT=512 → 0.64×),
2–4× SMEM (→ 0.29–0.54×; parity would need ~8×), dense FFN (experts=1, INT=512 → 0.939×, closest,
still 6.5% short). At *fixed small batch* F6 shows small per-layer wins (up to ~1.05×), but those sit
at low absolute throughput and never become the global best. The only fused strategy that wins
best-vs-best is the epilogue-only one above — a different strategy, not a lever on F6.

## Bottom line

- Unfused optimal batch = large (≥16384) → 1.104 Mtok/s. F6 optimal = small (≤~1568) → ~0.25 Mtok/s.
  Your directional hypothesis holds for F6.
- **F6's best throughput is 4.4–6.6× below unfused's**, because on-chip full-FFN fusion re-reads
  weights ∝ batch and forfeits amortization — the single structural reason.
- **But the best fused strategy isn't F6.** Epilogue-only fusion keeps the GEMMs weight-amortized,
  scales with batch, and slightly *beats* unfused (1.047×). So the right takeaway is: **fuse the cheap
  vector ops into weight-amortized GEMMs (scales); do NOT make the whole FFN intermediate-resident
  (caps throughput) unless you are deliberately running tiny batches.**

## Reproduce
```
conda run -n area python batch_sweep.py --gpu h100-sxm   # unfused / F6 / smart epilogue-only throughput
```
