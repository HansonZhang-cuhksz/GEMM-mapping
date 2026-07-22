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
