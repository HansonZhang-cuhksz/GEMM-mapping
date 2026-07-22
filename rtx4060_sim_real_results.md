# RTX 4060 sim–real fusion gap — results (task `RTX4060_SIM_REAL_TASK.md`)

> **CLOCKS LOCKED FOR THIS RUN.** The host locked the GPU to **1500 MHz core / 5501 MHz VRAM**
> (`nvidia-smi -lgc/-lmc`) — the exact calibration point of the `rtx4060-measured` profile —
> before this round. The lock was verified held under load (idle and sustained-GEMM samples all
> read 1500/5501), with one caveat: the 35 W power limit can still throttle *below* the lock
> under minutes of continuous max load (observed once, §5). The prior UNLOCKED-clock round is
> archived as `*_unlocked.*` and compared in §7.

**Date:** 2026-07-22 · **Device:** RTX 4060 Laptop GPU (8 GiB, 24 SM, cc 8.9, 99 KiB opt-in
SMEM/block, 32 MiB L2) · **Stack:** torch 2.11.0+cu130, triton 3.6.0, python 3.13, driver 610.53
(WSL2). **Raw data:** `rtx4060_measured.json` (T2+T3), `rtx4060_fusion.json` (T4; derived and
annotation fields under `annotations_post_hoc`), `notes/rtx4060_worklog.md` (step log, both
rounds). Adversarially reviewed against the raw JSONs; every figure exists in (or is derived by
a stated formula from) a deliverable.

---

## Verdict: **A — the fusion gains are real on working fused paths; the C500 null was a tooling problem.** With a quantitative caveat: the estimator gets the *direction* and *ranking* right but **over-predicts the magnitude ~2×** at these dims (delivered fraction ≈ 0.4–0.5).

At locked clocks, with 10 of 12 configs carrying a verified-fused path and 7 of 12 fully
drift-clean:

- Measured verified-fused gain geomean: **+2.9%** (all 10) / **+4.2%** (drift-clean 7), vs
  estimator **+6.9% / +8.6%** (stock ≈ adjusted at locked clocks). Per-config verified gains
  reach **+10.8%** (`swiglu_M8192_h1024`) and the clean residual config matches its estimate
  *exactly* (+5.2% measured vs +5.3% predicted, fused template at 1.0005× vendor speed).
- The genuinely-fused kernels run at ~vendor speed: hand Triton SwiGLU 1.03–1.08× the bare
  cuBLAS GEMM; forced Triton residual template 1.00× at the config that matches its estimate.
  Nothing resembling C500's 3.4× Triton tax (worst template case 1.18×).
- The C500-style "measure only what stock tooling gives you" view is now *positive* too
  (geomean 1.061 vs eager) — but the *verified-fused* metric is the honest one, and it says:
  real, positive, roughly half of predicted.

**Why the C500 saw nothing — the tooling story, per primitive:**

- **F1 residual:** out-of-the-box paths capture it on CUDA — `torch.addmm` +1.4…+2.1% on the
  two untainted configs, and both compile variants fuse via the vendor epilogue
  (profiler-verified). On the C500, addmm was 1% *slower*: a vendor-library difference.
- **F4 SwiGLU:** NO out-of-the-box path can fuse it even on NVIDIA — inductor refuses GEMM
  templates below 68 SMs, and even forced, SwiGLU is *structurally* unfusable as a template
  epilogue (§4). Only the custom dual-accumulator Triton kernel collects the gain
  (+2.4…+10.8% on 4 of 6 configs; ≈0 at h=4096 where the predicted gain is smallest).
- **Grouped MoE (the actual GLM decode regime):** no fused path exists on *either* stack — the
  estimator's prediction there (+7.9%) remains untestable without a custom grouped kernel.

| §5 criterion | observed (locked clocks) |
|---|---|
| `g_meas ≈ g_est`, fused kernel at ~vendor speed | Direction and ranking: yes; magnitude: measured ≈ 0.4–0.5× predicted. Fusion is genuinely faster (10-config verified geomean +2.9%, max +10.8%; fused kernels 1.00–1.08× vendor) → **A**, with the over-prediction caveat |
| `g_meas ≈ 0` despite competitive fused kernel | Only at the h=4096 extreme (predicted +4%, measured −0.7…−2.6% — the hand kernel's 1.06–1.08× overhead eats a small predicted gain) — not the across-the-board null the C500 showed |
| fused path itself slow (C500-style) | Never: worst fused path 1.18× vendor, vs C500's 3.4× |

---

## 1. Calibration context (T2, T3) — the profile reproduces at its calibration point

- **T2 peaks (locked):** 18.25 TF/s bf16 = **0.990×** the profile's 18.43; HBM 167.9 GB/s =
  **0.988×** the profile's 170. The `rtx4060-measured` profile is confirmed to ~1% at its own
  locked-clock operating point (implied sustained tensor clock 1485 MHz vs the 1500 lock). The
  T2-adjusted profile is therefore ≈ stock; both are carried through anyway.
- **T3 single-GEMM validation (24 shapes, locked):** geomean est/meas = **0.943** (stock) /
  0.952 (adjusted); **96% within 1.5×, 100% within 2×; ratios span 0.663–0.991** — in the prior
  0.72–0.96 band, and the estimator is now *uniformly* mildly optimistic (no pessimistic
  ratios; round 1's >1 ratios were boost-clock artifacts). FFN-stage subset: 0.948, 100% within
  1.5×. Every round-1 DVFS-flaky shape resolved (e.g. 2048×1024×1024: 13.9 TF/s in-sweep now).

## 2. What was measured (T4)

Per config, bf16, median of 30 CUDA-event samples after ≥15 warmup (the round-1 clock-warmer is
retained as a no-op guard; drift probes + per-config clock sampling verify the lock):

| path | what it is | fused? (profiler-verified) |
|---|---|---|
| `unfused` | eager: cuBLAS GEMM + separate elementwise kernel(s) | no (C500-comparable baseline) |
| `nocg` | compile max-autotune-no-cudagraphs: cuBLAS GEMM + ONE fused pointwise kernel | swiglu: no — **best unfused realization** (= the estimator's unfused model); residual: vendor-epilogue fused (4/4) |
| `compiled` | default `torch.compile(mode="max-autotune")` (cudagraphs on) | Triton-template: never (0/12). Residual: vendor-fused, but the cudagraph input-copies mean it beats eager in only 1/12 configs (10 strict losses, 1 tie) |
| `forced` | Triton GEMM template forced (`is_big_gpu` patch + TRITON-only backends, no cudagraphs) | residual: **yes** (single `triton_tem_fused_addmm`, 4/4); SwiGLU: **no** (structural, §4) |
| `triton` | hand dual-accumulator Triton GEMM+SwiGLU kernel, mini-autotuned | **yes** by construction (1 kernel); numerics pass vs fp32 reference on 6/6 |
| `addmm` | `torch.addmm` (cuBLASLt β-accumulate) | **yes** (no separate add kernel) |

`measured_gain_verified` = best *unfused* realization ÷ best *verified-fused* path (>1 ⇒ fusion
faster) — the verdict metric. `measured_gain` (vs eager, C500-convention) also stored.

## 3. Estimated vs measured fusion gain (locked clocks)

`est` = fusion_time_estimator at the same dims (stock profile; adjusted ≈ identical at locked
clocks, both in the JSON). `drift` = bare GEMM re-measured at config end ÷ start. `clk` =
sampled median graphics clock during the config.

**SwiGLU → up_gate epilogue (F4-analog, dense; verified path = hand Triton kernel). All six
configs drift-clean (0.98–1.02) at 1500 MHz:**

| config (M, h=inter) | est gain | measured (verified) | hand kernel ÷ vendor GEMM | delivered fraction |
|---|---|---|---|---|
| 2048, 1024 | 1.158 | **1.062** | 1.04× | 0.39 |
| 2048, 2048 | 1.079 | **1.024** | 1.07× | 0.30 |
| 2048, 4096 | 1.040 | 0.993 | 1.06× | −0.18 |
| 8192, 1024 | 1.159 | **1.108** | 1.08× | 0.68 |
| 8192, 2048 | 1.079 | **1.088** | 1.03× | 1.10 |
| 8192, 4096 | 1.041 | 0.974 | 1.08× | −0.63 |

**Residual → GEMM epilogue (F1-analog; verified paths = addmm and forced Triton template):**

| config | est gain | addmm gain | forced-template gain | forced ÷ vendor GEMM | drift / clk | reading |
|---|---|---|---|---|---|---|
| task dims M=2048 (n=16384,k=6144) | 1.053 | 1.021 | **1.052** | **1.0005×** | 1.00 / 1500 | ✔ clean — matches est exactly |
| task dims M=8192 | 1.053 | **1.014** | 0.934 | 1.13× | 1.06 / 1500 | template loses parity at these dims; addmm carries a smaller-than-predicted gain |
| GLM dims M=2048 (n=6144,k=16384) | 1.020 | 0.888 | 0.978 | 1.04× | 1.13 / 1500 | drift-tainted (power dips): contradicts both its round-1 value (1.006) and its M=8192 sibling — treat as unreliable |
| GLM dims M=8192 | 1.020 | **1.003** | 0.988 | 1.09× | 1.22 / **1290** | POWER-throttled below the lock (35 W cap) — order-confounded |

**MoE grouped-bmm rows (E∈{8,32}):** still no verified-fused path on this stack
(`measured_gain_verified` null). Their eager-vs-best-path gains (+3.0%, +13.4%) come from the
forced Triton *bmm* being a faster GEMM with a still-separate silu kernel — tooling speed, not
fusion. The estimator's grouped prediction (+7.9%) remains untestable.

**Aggregates (`annotations_post_hoc.aggregates`):**

| set | measured verified gain | est gain | delivered fraction (from geomeans) |
|---|---|---|---|
| all configs with a verified path, n=10 | **1.0285** | 1.0692 | **0.41** |
| drift-clean, n=7 (6 swiglu + residual_M2048) | **1.0419** | 1.0860 | **0.49** |
| swiglu only, n=6 (all clean) | 1.0402 | 1.0916 | 0.44 |
| all 12, C500-convention `measured_gain` vs eager | 1.0611 | — | — |

Per-row delivered fractions are noisy (−1.1…+1.1; per-row gains carry ±2%-ish measurement
noise against effects of similar size) — the geomeans are the robust statement.

## 4. Why the compiler paths cannot capture F4 (structural finding, unchanged from round 1)

`silu(gu[:, :inter]) * gu[:, inter:]` combines **two disjoint column-slices** of the GEMM
output — elements of two *different* output tiles. Inductor's template-epilogue fusion is
elementwise-on-own-tile only, so even forced it emits `triton_tem_fused_mm` + a separate
`triton_poi_fused_mul_silu_slice` kernel (`forced_kernel_evidence` in the JSON). cuBLASLt has
no SwiGLU epilogue in torch's binding. A true fused SwiGLU needs gate and up tiles computed in
the same CTA with dual accumulators — our 36-line hand Triton kernel does that, runs at
1.03–1.08× the vendor GEMM at locked clocks, and is the only path that collects the F4 gain.
(Split gate/up weights would make the epilogue single-tile and compiler-fusable, at the cost of
one extra `g` round-trip.)

**Where the ~2× over-prediction plausibly comes from** (hypotheses, consistent with the data):
(i) the estimator charges the eliminated activation read/write at DRAM bandwidth, but at the
small-h configs the `gu` tensor (8–67 MB for h ≤ 2048) is partly resident in the 32 MiB L2 when
the elementwise kernel runs, so the "eliminated traffic" was cheaper than modeled — the largest
single over-prediction is the smallest config (M=2048 h=1024: est +15.8% vs measured +6.2%,
9.6 points); (ii) at
h=4096 the predicted gain is small (+4%) and the hand kernel's 1.06–1.08× overhead vs cuBLAS
eats it — a kernel-engineering gap, not a model gap. The residual config where the fused
template hits exact vendor parity delivers its estimate exactly (+5.2% vs +5.3%), supporting
(ii).

**Numerics** (per-path evidence + `annotations_post_hoc.residual_numerics`): hand SwiGLU kernel
passes vs an fp32 reference on 6/6 configs (it is *more* accurate than eager — fp32 gate/up
through the silu). Residual rows' strict `numerics_ok=false` flags are false alarms (fused
rounds once vs eager twice; ≤1–2 bf16 ulp apart; on task-ordering dims the fused path is
strictly closer to fp32).

## 5. Caveats

- **The 1500/5501 lock held in 11/12 configs** (every ClockSampler median 1500). The exception:
  `residual_glm_M8192` sagged to a 1290 MHz median — on a 35 W part, `-lgc` caps the clock but
  cannot floor it against the power limit under minutes of continuous ~100 ms GEMMs. Its
  sibling `residual_glm_M2048` shows drift 1.13 for the same reason. Both flagged, treated as
  unreliable rather than evidence.
- The two h=4096 swiglu rows measure slightly negative verified gains against a small (+4%)
  prediction; they are honest datapoints (clean drift), not throttle artifacts — they mark the
  regime edge where the fusion benefit falls under the hand kernel's overhead.
- The estimator's over-prediction (~2× at these dims) is now cleanly measurable *because*
  clocks are locked; it was partially masked in round 1 by the unlocked memory clock (which
  shrank the predicted gains). This is a calibration finding about the vector-kernel/L2 model,
  not a refutation of the fusion direction — B ("gain ≈ 0 with working tooling") remains
  excluded by the 10-config +2.9% geomean and the +5.2/+10.8% star configs.
- 8 GiB forces scaled/dense configs; grouped-MoE covered qualitatively only.

## 6. Answers to §6's checklist (locked clocks)

| quantity | C500 (measured) | **RTX 4060 (locked 1500/5501, measured here)** |
|---|---|---|
| per-fusion gain, decode-analog | ≈0% / negative | verified-fused geomean **+2.9%** (n=10) / **+4.2%** (drift-clean n=7); per-config −2.6%…**+10.8%** vs est +2.0%…+15.9%; ≈0.4–0.5× the predicted magnitude |
| F1 residual via addmm | 1% *slower* | **+1.4…+2.1% faster** (untainted configs); forced Triton template **+5.2% at 1.0005× vendor** on the clean config — exactly the estimate |
| F4 SwiGLU via compile | no fuse | no fuse either — SM-count gate AND structural (§4). Hand fused kernel: **+2.4…+10.8%** on 4/6 configs, ≈0 at h=4096 |
| Triton fused GEMM vs vendor | 3.4× slower | hand kernel **1.03–1.08×**; forced templates **1.00–1.18×** — the C500 fusion tax does not generalize |
| single most informative datapoint | — | max-autotune alone does **not** produce a fused-and-winning Triton GEMM — but hand-written fused Triton GEMMs run at ~vendor speed and win where the predicted gain exceeds their small overhead |

## 7. Locked vs unlocked clocks (round 2 vs round 1, archived as `*_unlocked.*`)

| quantity | unlocked (R1) | **locked 1500/5501 (R2)** |
|---|---|---|
| T2 peak TFLOP/s | 18.79 (1.019× profile; boost) | **18.25 (0.990×)** |
| T2 HBM GB/s | 214.8 (1.264×; mem clock ~7 GHz) | **167.9 (0.988×)** |
| T3 geomean est/meas | 0.937 (max ratio 1.148; 95.8% within 2×) | **0.943 (max 0.991; 100% within 2×)** |
| T4 drift-clean configs | 5/12 (DVFS + thermal throttle) | **7/12** (residual power-cap dips remain) |
| T4 verified gain geomean (all with path) | 0.983 (throttle-confounded rows dragged it) | **1.0285** |
| verified gain, clean set | +3.7% vs est +4.6% (adjusted) | **+4.2% vs est +8.6%** |
| `swiglu_M8192_h2048` verified | 0.883 (throttle order-confound, excluded) | **1.088 vs est 1.079** — vindicates the R1 exclusion |
| flaky small shapes | 2048×1024×1024 at 2.2–5.8 TF/s in-sweep | clean 13.9 TF/s |

Two lessons: (1) every round-1 anomaly attributed to DVFS resolved under the lock — the
methodology calls (clock warmer, drift probes, clean-set restriction) were correct; (2) the
estimator's predicted gains are bandwidth-sensitive (unlocked mem clock shrank them), so the
locked run is the apples-to-apples test of the calibrated profile — and it shows the ~2×
magnitude over-prediction that the unlocked round could not cleanly separate.

**Actionable for C500:** unchanged, with sharper expectations — a custom MACA-CUTLASS
fused-epilogue kernel (dual-accumulator for SwiGLU; grouped for the real MoE layer) should
recover a real but likely-half-of-estimated gain, worth it at the +5–11% configs (small
hidden/inter, residual epilogues at vendor parity) and marginal at large-h dense FFN dims.

---

# T6 — RMSNorm + MoE-structure fusion tests (A–E), LOCKED CLOCKS

> Clocks locked **1500 MHz core / 5501 MHz VRAM** by the host throughout (verified before the
> run; ClockSampler medians 1500 on most rows). The 35 W power cap still dips below the lock on
> the heaviest sustained configs (A-epilogue rows show medians down to 1140 MHz; flagged by the
> per-config drift probes). Raw data: `rtx4060_rmsnorm.json` (A), `rtx4060_router_prologue.json`
> (B), `rtx4060_fusion.json` (C rows `router_topk`, D rows `ffn_levels`/`ffn_grouped`, E rows
> `merge_r2f`, plus top-level `router_topk_drop_evaluation` and `merge_token_independence`).
> Host decisions (asked, not assumed): A-epilogue prefill capped at 32768; A-epilogue input
> includes a pre-materialized residual (layer-faithful `RMSNorm(mla_o(x)+res)`); all five tests
> run. Per T6.0.4, structure-blind estimator cells (A-epilogue, C) are excluded from every
> est-vs-measured aggregate; all quoted numbers below are **measured** unless tagged est.

## T6.A — RMSNorm into mla_o (epilogue) + prologue fallback

**Epilogue (cross-tile reduction over N): structurally infeasible — confirmed on hardware.**
0/7 stock-path fusions (compiled/nocg/forced all leave a separate reduction kernel;
profiler-verified; `addmm` N/A — no reduction epilogue in cuBLASLt's binding). The hand
wide-tile kernel (option 1, BM=16) failed at all 7 configs — a Triton `CompilationError`:
`tl.arange` requires a power-of-2 width, so the `[BM, 6144]` full-row tile cannot even be
expressed; padding N to 8192 would need a `16·8192·4 = 512 KiB` fp32 stage, and the spec's own
budget arithmetic (`16·6144·4 = 384 KiB` vs 99 KiB, recorded per-row in `smem_budget_note`)
already rules the design out regardless of expression. With no
fusable path, measured gains sit at 0.99–1.01 (n=7, M=512…32768). The structure-blind est
cell (`est=1.033 INVALID`) is the recorded model-vs-hardware mismatch: a tile-local traffic
model predicts a fuse the machine cannot express. This is the same lesson as T4-§4, one level
harder: SwiGLU needed 2 column-slices; RMSNorm needs the whole row.

**Prologue (reduction over K, tile-local): fusable only by the hand kernels — and the payoff
is host-dependent.** P2 (sumsq pass → GEMM with scaled-A-load prologue) passes fp32 numerics
on 10/10 configs (P1 on 9/10 — its two-pass variant misses the tolerance at tpe=1024, rel
0.062 vs 0.051); no stock path removes the normalized-A round-trip (0/10, verified). Measured
verified gains vs the honest 2-kernel P2 estimate (unfused baseline that physically writes
normalized x):

| host | configs | measured verified gain | honest-P2 est | reading |
|---|---|---|---|---|
| up_gate `[tpe, 4096, 6144]` | tpe 16…4096 | **0.74…1.01** (drift-clean geomean 0.87, n=5) | +0.4…+2.6% | hand GEMM cannot match cuBLAS on wide shapes; the small predicted saving is eaten by the GEMM-quality gap |
| router `[M, 256, 6144]` | M ∈ {2048, 32768} | **+26.8% / +17.1%** | +27.4% / +29.6% | ✔ on the skinny host the prologue saving is large and the hand kernel is competitive — matches the estimate |

## T6.B — router prologue (b2 residual; b1 residual+RMSNorm)

**The largest verified gains of the whole study.** The router GEMM (`n=256`) is dominated by
reading `[M, 6144]` inputs, so folding the residual (b2) — and additionally the RMSNorm (b1,
gamma pre-folded into `Wp = Wr·diag(gamma)`, untimed) — into the A-load removes entire passes
over the hidden state:

| variant | drift-clean gains (M ≥ 2048) | geomean | est geomean | fused path |
|---|---|---|---|---|
| b2 residual | 1.27 → 2.17 (generally grows with M; peak at M=16384) | **1.844** | 2.230 (est) | hand kernel only |
| b1 +RMSNorm | 1.13 → 1.77 | **1.478** | 3.037 (est) | hand kernel only (single-pass co-accumulated sumsq, no split-K) |

Inductor's forced template folded the input add **0/16** — no stock path fuses either variant;
the hand kernel is the only fused realization (b1's K-reduction is un-synthesizable by any
compiler on this stack, per B.3). The b1 M=131072 row (measured gain 9.79×) is excluded: its
eager baseline ran 294 ms vs a 23.7 ms GEMM — a memory-pressure-degraded baseline at ~6.0 GiB
of buffers, not a fusion effect. Two mandated caveats: the router is a minor cost center
(~16× below one up_gate expert-layer in FLOPs), and in the real layer `h`/`x` are shared
downstream, so the router-attributable saving is only the avoided re-read of `x` — smaller
than this standalone microbenchmark shows. The estimator over-predicts (delivered ≈ 0.83 of
est for b2, ≈ 0.5 for b1) mainly because its optimal-mapping fused GEMM is faster than a real
skinny-N kernel — but the direction and the order of magnitude are confirmed.

## T6.C — top-k as the router-GEMM epilogue: attempt-and-DROP (condition met)

Per the C.6 protocol (`router_topk_drop_evaluation`): **(a)** no stock path fused on 9/9 —
compiled/nocg/forced all leave a distinct topk/sort kernel (recorded evidence; top-k is a
cross-tile selection outside the pointwise/broadcast epilogue vocabulary of
cuBLASLt/CUTLASS/inductor); **(b)** the one viable custom route (BN=256 full-row-resident
Triton kernel, feasible on this SMEM, numerics-verified value+index-set) never beat
GEMM+torch.topk: measured gains 0.75–1.00 on all 6 drift-clean configs (the drift-tainted
M=512 row reads 0.58). The estimator's traffic-only bound (recorded cells +1.3…+1.9%; spec
a-priori ceiling ~3.5% — flagged NOT-a-prediction) did not materialize — the full-row
accumulator's GEMM-efficiency tax exceeds the fixed logit-round-trip ceiling. **Test C is
dropped from the gains tables as specified; the raw rows + profiler evidence remain in the
JSON for audit.** (The host's stated doubt is confirmed.)

## T6.D — MoE FFN levels L1/L2/L3 at GLM per-expert dims

**L2 (SwiGLU into up_gate at `[tpe, 4096, 6144]`): neutral-to-negative in the per-expert
regime.** Isolated up+SwiGLU verified ratios `r_up_swiglu` = 0.74–1.10 across tpe 16…4096;
only tpe=64 wins (+10.0% vs isolated est +2.0%); full-FFN `r_L2` = 0.80–1.07 vs est
1.003–1.022. This
refines T4: SwiGLU fusion paid on dense mid-size shapes (M=2048/8192), but at the skinny-tpe
GLM expert shapes the hand kernel (and the forced template) cannot beat cuBLAS by enough to
collect the small predicted saving. Grouped-bmm cross-check (E=8, tpe∈{64,512}): grouped runs
0.80–0.97× of 8× single-expert (occupancy benefit); grouped `r_L2` 0.97/1.02 — consistent.
(The grouped rows' fp32-reference numerics field is invalid — a reference-construction bug
(rel ~10³, flagged in the worklog); the timing ratios are unaffected.)

**L3/F6 (whole FFN in one kernel): infeasible in Triton on this hardware — estimator-only.**
Evidence recorded per D.4 drop rules: the 4-chunk formulation needs 176–322 KiB SMEM
(Required, from `OutOfResources`) vs 101376 B because Triton stages every `tl.dot` operand
with no cross-phase buffer reuse (the paper budget was 84 KiB); an 8-chunk variant fits SMEM
but miscompiles (corrupt output, rel 57–252, while the identical phase-1 chunk in isolation
is correct at rel 0.011) — a Triton SMEM-liveness failure with 8 persistent dot operands. The
tpe-parametrized estimator (`estimate_ffn_fused_m`, acceptance-checked: reproduces 0.2591× at
m=64, scales with mt, m0 SMEM-capped at 16) stands as the prediction: **est** `r_L3` = 1.005×
at `mt=1` (tpe=16) collapsing to 0.15× (~6.7× slower) for `tpe ≥ 256` — the `mt×`
weight-re-read cliff that makes F6 a SKIP at deployment batch sizes. A CUTLASS-class kernel
with explicit SMEM management would be required to test it on silicon.

## T6.E — residual2 into the expert-merge (r2f): the cleanest confirmation in the study

Stock `torch.compile` **fuses this out of the box** (8/8 measured configs: one
`triton_per_fused_add_mul_sum_unsqueeze` kernel absorbs mul+sum+residual-add;
profiler-verified; `forced` N/A — no GEMM to template; `addmm`-analog `baddbmm` attempted per
spec — never the fastest fused path (it beats only the cudagraph-taxed default compile),
dropped where >1.5× the reduction):

| tokens | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 | 49152 |
|---|---|---|---|---|---|---|---|---|
| verified gain | 1.250 | 1.202 | 1.248 | 1.253 | 1.242 | 1.284 | 1.232 | 1.230 |
| stock-compile-only gain | 1.250 | 1.128 | 1.197 | 1.231 | 1.226 | 1.284 | 1.232 | 1.230 |

**est = 12/10 = 1.20 exactly — profile- AND token-independent** (both sides pure traffic/bw;
stock ≡ adjusted). Measured delivered gain ≈ 1.20–1.28, i.e. ~100%+ of the predicted ratio at
every size, drift-clean geomean **1.246 verified / 1.218 stock-only** (n=6). The 131072 point
is arithmetically infeasible on 8 GiB (`expert_outs` alone = 12.9 GB, dropped with the
arithmetic recorded) and is **defensibly covered by the token-independence assertion**:
verified gains at the mandated 512/8192/32768/49152 points are 1.2497/1.2423/1.2318/1.2298 —
spread 1.6% ≤ 5% → `merge_gain_token_independent: true` (`merge_token_independence` in the
JSON). Measurement notes: the 49152 point OOMed at end-of-sweep and exposed an int32
pointer-arithmetic overflow in the hand kernel (flat indices > 2³¹ — fixed to int64); it was
re-measured in a fresh process, where its *eager* paths paged into host memory (12.4 s, WSL2
oversubscription) — its gain is computed against the clean device-resident 2-kernel baseline
(45.3 ms vs 36.9 ms fused). This is the strongest verdict-A datapoint: a memory-bound fusion
the estimator prices exactly, captured by stock tooling with **no custom kernel** — precisely
the class of win the C500's stack could not collect.

## T6 verdict — how the five tests sharpen the study

The realizability hierarchy is now measured end-to-end. **Fusions win where (1) the epilogue/
prologue is tile-local AND (2) the fused kernel matches vendor GEMM quality (or there is no
GEMM at all):** expert-merge +13…+28% out of the box via stock compile (verified-path gains
+20…+28% including the optional hand kernel) (E), router prologue +27%…+117% via
hand kernels (B, A-router). **Fusions are neutral-to-negative where cuBLAS's lead on the host
GEMM exceeds the traffic saving** (SwiGLU/RMSNorm-prologue at wide per-expert shapes — D, A).
**Fusions are structurally unreachable — by any stock path, and sometimes by any Triton
kernel — where the epilogue crosses tiles**: RMSNorm-over-N (A), top-k selection (C), and the
two-GEMM F6 chain (D). The estimator's tile-local traffic model predicts the *direction*
correctly wherever its structural assumptions hold and over-predicts magnitude by ~1.2–2×
(kernel-quality gap), but it is **structure-blind** — its A-epilogue and top-k "wins" are
artifacts its own `Epilogue` vocabulary cannot see, now flagged as such in every table. For
the C500: the recoverable wins are the E-class (any working compiler) and B-class (custom
prologue kernels); the D/F6-class needs CUTLASS-grade SMEM control on any vendor.
