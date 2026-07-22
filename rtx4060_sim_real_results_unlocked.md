# RTX 4060 sim–real fusion gap — results (task `RTX4060_SIM_REAL_TASK.md`)

**Date:** 2026-07-21 · **Device:** RTX 4060 Laptop GPU (8 GiB, 24 SM, cc 8.9, 99 KiB opt-in
SMEM/block, 32 MiB L2) · **Stack:** torch 2.11.0+cu130, triton 3.6.0, python 3.13, driver 610.53
(WSL2) · **Clocks:** UNLOCKED (no root; task-§7 fallback: per-config nvidia-smi sampling + a
busy-GEMM clock warmer before every timed region — see §5).
**Raw data:** `rtx4060_measured.json` (T2+T3), `rtx4060_fusion.json` (T4; derived/annotation
fields under `annotations_post_hoc`), `notes/rtx4060_worklog.md` (step log).
This writeup was adversarially reviewed against the raw JSONs; every figure below exists in (or
is derived by a formula stated here from) a deliverable file.

---

## Verdict: **A — the fusion gains are real and the estimator's predictions are approximately right; the C500 null was a tooling problem.** (Scope: demonstrated in the dense regime; the grouped-MoE regime is unfalsifiable on any current stack — see below.)

On the 5 drift-clean configs (of 12; criterion |drift−1| ≤ 0.05, see §3) the predicted fusion
gain **materializes**: measured verified-fused gain geomean **+3.7%** vs predicted **+4.6%**
(T2-adjusted profile; +5.7% stock), with per-config delivered-fraction geomean ≈ 0.78 (range
0.37–1.74). The genuinely-fused kernels run at **~vendor-GEMM speed**: hand Triton SwiGLU kernel
0.98–0.99× the bare cuBLAS GEMM on clean configs; forced Triton residual template 1.00–1.01×. Nothing
resembling the C500's 3.4× Triton fusion tax exists on this stack (worst inductor-template case:
1.59×).

**Why the C500 saw nothing — the tooling story, sharpened by primitive:**

- **F1 residual:** out-of-the-box paths DO capture it on CUDA — `torch.addmm` wins +0.6…+1.2%
  on all 4 configs and even default `torch.compile` fuses it into the vendor epilogue
  (profiler-verified: no separate add kernel). On the C500, addmm was 1% *slower* — a genuine
  vendor-library difference, i.e. tooling.
- **F4 SwiGLU:** NO out-of-the-box path can fuse it, even on NVIDIA — inductor refuses Triton
  GEMM templates below 68 SMs, and even when forced, SwiGLU is *structurally* unfusable as a
  template epilogue (§4). Only our custom dual-accumulator Triton kernel collects the gain
  (+5.6…+7.4% on clean configs). The C500 equivalent would be a custom MACA-CUTLASS kernel.
- **Grouped MoE (the actual GLM decode regime):** no fused path exists on *either* stack (hand
  kernel is dense-only; inductor's bmm template can't fold SwiGLU) — the estimator's predicted
  +6.4…+7.9% there is untested and untestable without a custom grouped kernel.
- If one only measures what stock tooling delivers (the C500 methodology), the null reproduces
  on NVIDIA too: C500-convention `measured_gain` geomean across all 12 configs is **0.992**.

| §5 criterion | observed |
|---|---|
| `g_meas ≈ g_est`, fused kernel at ~vendor speed | **Yes** on the 5 drift-clean configs (measured +3.7% vs adjusted-est +4.6% geomean; fused kernels 0.98–1.01× vendor) → **A** |
| `g_meas ≈ 0` despite competitive fused kernel | No on the 5 clean configs; 1 config (`swiglu_M2048_h1024`) has an unmeasurable baseline (see §3) and is no-data, not a B-signature |
| fused path itself slow (C500-style) | Never at C500 scale. Worst inductor template 1.59× vendor; the hand kernel stayed at 1.01–1.05× even in the throttled configs (re-referenced to the contemporaneous GEMM) |

---

## 1. Calibration context (T2, T3)

- **T2 peaks:** 18.79 TF/s bf16 GEMM = **1.019×** the profile's 18.43 (implied sustained tensor
  clock 1529 MHz vs the locked-1500 calibration); HBM 214.8 GB/s = **1.264×** the profile's 170
  (unlocked memory clock ~7 GHz vs locked 5501 MHz). An adjusted profile
  (`clock_hz=1.529e9, bw=2.148e11`) is carried through all estimator comparisons below.
- **T3 single-GEMM validation (24 shapes):** geomean est/meas = **0.937** (stock) / 0.919
  (adjusted); **96% of shapes within 1.5×, 95.8% within 2×** (the one outlier either way is
  2048×1024×1024 at ratio 0.327, a DVFS-flagged shape that measures 13.4 TF/s standalone) —
  inside the prior 0.72–0.96 calibration band. The T4 regime (up_gate / mla_o shapes) sits at
  ratios 0.88–1.15. The estimator's absolute scale is trustworthy here, so its *relative*
  fusion predictions are the right yardstick.

## 2. What was measured (T4)

Per config, all timed bf16 with median-of-30 CUDA events after 15 warmup + clock warmer:

| path | what it is | fused? (profiler-verified) |
|---|---|---|
| `unfused` | eager: cuBLAS GEMM + separate elementwise kernel(s) | no (C500-comparable baseline) |
| `nocg` | `torch.compile` max-autotune-no-cudagraphs: cuBLAS GEMM + ONE fused pointwise kernel | swiglu: no — **best unfused realization** (= the estimator's unfused model); residual: vendor-epilogue fused |
| `compiled` | default `torch.compile(mode="max-autotune")` (cudagraphs on) | Triton-template: never (0/12). Residual: vendor-epilogue fused, but its cudagraph input-copies make it lose to eager in 10/12 configs and to the best unfused realization in 11/12 |
| `forced` | inductor Triton GEMM template forced (`is_big_gpu` patch + TRITON-only backends, no cudagraphs) | residual: **yes** (single `triton_tem_fused_addmm` kernel, 4/4); SwiGLU: **no** (structural, §4) |
| `triton` | hand dual-accumulator Triton GEMM+SwiGLU kernel, mini-autotuned (6 tile configs) | **yes** by construction (1 kernel) |
| `addmm` | `torch.addmm` (cuBLASLt β-accumulate) | **yes** (no separate add kernel; the DtoD copy of `res` is the β-accumulate input) |

`measured_gain_verified` = best *unfused* realization ÷ best *verified-fused* path (>1 ⇒ fusion
faster) — the verdict metric. `measured_gain` (vs eager, C500-convention) is also in the JSON.

## 3. Estimated vs measured fusion gain

`est` = fusion_time_estimator at the same dims, stock / T2-adjusted profile. `drift` = bare GEMM
re-measured at config end ÷ start (≈1.00 = clean; the probe was added before the full run for
exactly this purpose; the ±0.05 clean threshold was fixed at analysis time). `late-phase gain` =
best-unfused ÷ fused where **both** operands were measured after the compile/autotune phase
(swiglu: `nocg/triton`; residual: `nocg/forced`; stored in
`annotations_post_hoc.derived_late_phase`) — drift-robust within each config.

**SwiGLU → up_gate epilogue (F4-analog, dense; verified path = hand Triton kernel):**

| config (M, h=inter) | est stock | est adj | measured (verified) | late-phase gain | fused kernel ÷ contemporaneous GEMM | drift | reading |
|---|---|---|---|---|---|---|---|
| 2048, 1024 | 1.158 | 1.127 | — | (1.06) | 1.09× | 0.86 | **baselines mutually inconsistent** (bare GEMM 0.556→0.478 ms across the config; eager unfused 0.458 < both — impossible at steady clock): its late-phase field (1.06, pro-fusion) and repeat-referenced ratio (1.09×, anti) disagree → no usable datapoint either direction |
| 2048, 2048 | 1.079 | 1.064 | 1.046 | 1.046 | 1.08× | 0.87 | drift-tainted (excluded from aggregate); late-phase agrees with est |
| 2048, 4096 | 1.040 | 1.032 | **1.056** | 1.056 | **0.99×** | **1.00** | ✔ clean, confirms (above est) |
| 8192, 1024 | 1.159 | 1.128 | **1.074** | 1.074 | **0.98×** | **1.01** | ✔ clean, confirms (58% of predicted) |
| 8192, 2048 | 1.079 | 1.064 | 0.883 | 0.890 | 1.006× | 1.35 | throttle order-confound (see §5): excluded; note the fused kernel itself held vendor parity |
| 8192, 4096 | 1.041 | 1.034 | 0.832 | 0.940 | 1.047× | 1.23 | same |

**Residual → GEMM epilogue (F1-analog; verified paths = addmm and forced Triton template):**

| config | est stock | est adj | addmm gain | forced-template gain | forced ÷ contemporaneous GEMM | drift |
|---|---|---|---|---|---|---|
| task dims M=2048 (n=16384,k=6144) | 1.053 | 1.043 | 1.012 | 1.036 | 1.00× | 1.14 (excluded) |
| task dims M=8192 | 1.053 | 1.043 | 1.008 | **1.035** | **1.00×** | **1.00** ✔ |
| GLM dims M=2048 (n=6144,k=16384) | 1.020 | 1.016 | **1.006** | 1.005 | 1.01× | **1.00** ✔ |
| GLM dims M=8192 | 1.020 | 1.016 | 1.007 | **1.015** | **1.00×** | **1.00** ✔ |

**Aggregates (all in `annotations_post_hoc.aggregates`):**

| set | measured verified gain (geomean) | est stock | est adj |
|---|---|---|---|
| drift-clean, n=5 (the verdict set) | **1.0368** | 1.0569 | 1.0461 |
| clean-5 + the two drift-excluded *positive* rows (1.046 at drift 0.87, 1.036 at drift 1.14), n=7 | 1.0379 | — | 1.0481 |
| every drift<1.05 config incl. the unmeasurable-baseline row (0.879), n=7 | 1.0139 | — | 1.0598 |
| all configs with a verified path, n=10 | 0.9825 | — | — |
| all 12, C500-convention `measured_gain` | 0.992 | — | — |

Transparency: the verdict rests on 5 of 12 configs; the excluded 7 are 1 inconsistent-baseline
row, 1 drift-0.87 row and 1 drift-1.14 row (both *positive*, 1.046/1.036 — the symmetric rule
costs the verdict evidence too), 2 throttle-confounded rows, and 2 MoE rows with no verified
path. The sensitivity rows above bracket the reading: adding back the tainted-but-plausible
positive rows leaves the result unchanged (+3.8% vs +4.8%); the only cut that erodes it to
+1.4% vs +6.0% is the one that counts the unmeasurable-baseline row's 0.879 as evidence. The
per-row late-phase gains (1.005–1.074 on every non-throttled row) support the clean-set reading.

**MoE grouped-bmm rows (E∈{8,32}):** no verified-fused path exists on this stack (hand kernel
dense-only; inductor bmm template cannot fold SwiGLU either) — `measured_gain_verified` is null.
Their unfused-vs-compiled gains (0.99 / 0.81, the latter with drift 1.28) measure tooling, not
fusion. This is the *actual* GLM decode regime: a production fix needs a custom grouped fused
kernel on any vendor.

## 4. Why the compiler paths cannot capture F4 (structural finding)

`silu(gu[:, :inter]) * gu[:, inter:]` combines **two disjoint column-slices** of the GEMM output
— elements from two *different* output tiles. Inductor's template-epilogue fusion is
elementwise-on-own-tile only, so even with templates forced it emits `triton_tem_fused_mm` + a
separate `triton_poi_fused_mul_silu_slice` kernel (profiler evidence in
`rtx4060_fusion.json:forced_kernel_evidence`). cuBLASLt has no SwiGLU epilogue in torch's
binding. A true fused SwiGLU needs a kernel that computes the gate tile and the up tile in the
same CTA with dual accumulators — our 36-line hand Triton kernel does exactly that, runs at
0.98–0.99× the vendor GEMM on clean configs, and is the only path that collects the predicted
gain. (Split gate/up weights would make the epilogue single-tile and compiler-fusable at the
cost of one extra `g` round-trip — an implementation avenue for stacks without custom kernels.)

The C500's "3.4× Triton fusion tax" does not generalize, but the tax is shape-dependent even on
NVIDIA: forced inductor GEMM templates run **0.96–1.59×** the vendor GEMM here (1.59× on the
drift-clean `swiglu_M8192_h1024`); it is the *hand* kernel that consistently reaches ~1.0×.

**Numerics** (per-path evidence in the JSON; post-hoc analysis in
`annotations_post_hoc.residual_numerics`): the swiglu hand kernel is *more* accurate than the
eager reference vs an fp32 ground truth on 5/6 configs (rel-max 0.004–0.022 vs eager's
0.015–0.022; equal on the sixth) because it keeps gate/up in fp32 through the silu. The residual
rows' strict `numerics_ok=false` flags are false alarms: addmm/fused round once where eager
rounds twice; on task-ordering dims the fused path is strictly closer to fp32 (rel-max 0.004 vs
0.012–0.015) with 99.8% of elements within 1 bf16 ulp of eager; on GLM-ordering dims (k=16384)
*both* paths carry comparable bf16 accumulation error from cancellation (rel-max ≈1.07–1.35 both
ways on near-zero outputs) and agree with each other to within max-abs 4.0 at |values| up to
~512 (≤2 bf16 ulp; 88% of elements within 1).

## 5. Caveats and methodology honesty

- **Clocks could not be locked** (WSL2, no root). Mitigations: continuous busy-GEMM warmer
  before every timed region (without it, short kernels measure at idle 210 MHz — up to 7×
  wrong; T3's band went 0.768 → 0.937 with it), per-config clock sampling, per-config drift
  probes. Residual risk is the in-config ordering: baseline paths are measured minutes before
  the fused paths (autotune in between), so any drift lands asymmetrically on the fused side.
- **Thermal throttling** on this 35 W part: sustained clocks sag 1725 → 1140 MHz across the run.
  In the two throttled swiglu configs the *fused-path* times run 26–39% over the estimator while
  the earlier-measured unfused paths run only 3–11% over — an ordering artifact, not fused-path
  slowness (the hand kernel is 1.006×/1.047× the *contemporaneous* GEMM there). They are
  excluded as order-confounded, **not** counted as §5-criterion-3 evidence.
- The estimator over-predicts the realized gain on clean configs (delivered fraction geomean
  0.78, range 0.37–1.74; the M=8192 h=1024 row delivers 58% of predicted). This is consistent
  with its T3 absolute bias (geomean 0.94) and the unlocked memory clock making vector kernels
  ~26% cheaper than the stock profile assumes — but at n=5 a ~20% structural over-prediction of
  fusion benefit cannot be excluded. What *can* be excluded is verdict B's "gain ≈ 0 with
  working tooling": every clean config shows a real, positive, roughly predicted-magnitude gain.
- 8 GiB forced scaled/dense configs; the full GLM MoE layer was not run (grouped regime covered
  qualitatively only — no fused path exists to measure).

## 6. Answers to §6's checklist

| quantity | C500 (measured) | **RTX 4060 (measured here)** |
|---|---|---|
| per-fusion gain, decode-analog | ≈0% / negative | **+3.7% geomean verified on drift-clean configs** (per-config +0.6%…+7.4%; vs est +1.6%…+12.8% adjusted, +2.0%…+15.9% stock). All-configs stock-tooling view: 0.992 (the C500 null reproduces without custom kernels) |
| F1 residual via addmm | 1% *slower* | **+0.6…+1.2% faster** (4/4 configs); forced Triton template +0.5…+3.6% at 1.00× vendor speed |
| F4 SwiGLU via compile | no fuse | no fuse either — SM-count gate AND structural (§4). Hand fused kernel: **+5.6…+7.4%** on clean configs at 0.98–0.99× vendor speed |
| Triton fused GEMM vs vendor | 3.4× slower | inductor templates **0.96–1.59×** (shape-dependent); hand fused kernel **~0.99×** (clean configs) |
| single most informative datapoint | — | max-autotune alone does **not** produce a fused-and-winning Triton GEMM — but the *hand-written* fused Triton GEMM both (a) runs at vendor speed and (b) beats the best unfused path by +5.6…+7.4% (clean configs) |

**Actionable for C500:** verdict A says the win is recoverable — but only via a custom
fused-epilogue kernel (MACA-CUTLASS), a dual-accumulator design for SwiGLU, and for the real
MoE decode layer it must be a *grouped* fused kernel (a gap that exists on NVIDIA today too).
