# Fusion analysis on the physical MetaX C500 — measured vs snowcat-roofline estimator

We ran the fusion studies (until now all analytical) on a real **MetaX C500** and compared to the
snowcat estimator configured with a measured C500 profile. Measurement (`metax_measure.py`, bf16,
cuda-events) in the `fusion` env; estimator + C500 model (`metax_c500_model.py`, `metax_compare.py`)
in `area`. Verified by a 5-agent workflow incl. an adversarial audit (methodology + calibration
sound; one root-cause correction). Log: `notes/metax_c500_plan.md`.

## Measured C500 profile (vs the H100 estimator profile)

| | MetaX C500 (measured) | H100 (est. profile) |
|---|---|---|
| Peak BF16 | **226 TFLOP/s** (measured 229) | 989 |
| HBM BW | **1.43 TB/s** (measured 1.44) | 3.35 |
| SMs | 104 | 132 |
| L2 | **8 MiB** | 50 |
| SMEM/block | **64 KiB** | 227 |
| ridge OI | 158 FLOP/B | 295 |

Model reproduces the hardware on every measured axis; the two uncalibrated guesses (400 ns latency,
`bw_saturation_sms=20`) are **provably immaterial** — sweeping latency ∈ {200,400,800} ns and
bw_sat ∈ {5,20,40} leaves every verdict byte-identical (all sampled GEMMs saturate the 104 SMs, so
those terms never bind).

## Estimator accuracy on real C500: order-of-magnitude right, uncalibrated

Over 32 unique GEMMs, **est/meas geomean 1.05, median 1.15, 56% within 1.5×, 81% within 2×.** Two
one-directional biases:

1. **Over-predicts compute-heavy GEMMs 2.5–2.8×** — `4096×4096×16384` (2.51, meas 207 TF/s),
   `mla_o 2048×6144×16384` (2.58, 213), `8192×4096×16384` (2.77, 228); also `8192³` (1.27),
   `16384³` (1.77). The estimator calls them memory-bound; the real C500 runs them near peak.
2. **Under-predicts small/narrow GEMMs ~1.5×** — `1024³` (0.42), `glm down` (0.43),
   `multi_stage w=128` (0.52).

**Corrected root cause (the audit overturned my first guess):** the over-prediction is **not** the
8 MiB L2 traffic model — it's the **SMEM-coupled tile cap.** C500's 64 KiB SMEM + the mandatory
C≥2 double-buffer force the optimal-tile search down to `128×64×32`, whose snowcat re-read
multipliers put modeled OI at ~90–124 (below the 158 ridge), so genuinely compute-bound GEMMs get
labelled memory-bound. Raising L2 to 50 MiB does **not** fix `4096×4096×16384`; the matching
compute-bound tile (`256×256×16`, OI 130) is rejected as `fits_smem=False`. The real fix is what
production kernels do: **decouple the reuse/OI-setting register output tile (BM×BN, backed by ~128k
regs/SM) from the small double-buffered SMEM staging** — a limitation of the estimator's tile model
on small-SMEM hardware. The under-prediction is a missing **~20 µs launch overhead + narrow-N
tensor-core inefficiency**.

**Empirical calibration** (for the measured set): effective L2 → 32 MiB (numerically caches
operand A) + 20 µs launch overhead → **geomean 0.99, 100% within 1.5×, max 1.30×.** Kept OUT of the
model file: the L2 bump is a numerical compensation for the tile-cap bug, not the true C500 L2
(8 MiB is real) — documenting rather than hard-coding it keeps the profile honest.

## Fusion verdicts: C500 vs H100 — tighter hierarchy ⇒ more infeasible, fewer wins

| study | H100 | C500 | why |
|---|---|---|---|
| 27-shape chain | 3 / 19 / 5 | **1 / 5 / 21** | held-slice cap 6752→1536 wide (64 vs 227 KiB SMEM) → infeasible 4× |
| focused flash-attn (24) | 12 / 12 / 0 | **6 / 18 / 0** | weights L2-resident only to K1≈614 vs 3840 (eff-L2 4.8 vs 30 MiB) |
| multi-GEMM L=6 speedup | w128 4.59×, w256 2.30×, w512 1.16× | **2.46× / 1.23× / 1.00×**; **w1024 INFEASIBLE** | lower peak drives feasible chains to the compute floor |
| SMEM N* (full sched) | 12 / 6 | **2** | 64 KiB SMEM fills far sooner |

## Which C500 verdicts to trust (margin vs the estimator's own ~1.3–1.8× error)

- **TRUSTED — FUSE the narrow chains** (`multi_w128` L3/L6, **2.46×**): margin dwarfs the error,
  mechanism (removing 32–64 MiB intermediate round-trips that spill the 4.8 MiB L2) is physical, and
  the unmodeled launch-overhead saving only makes the real win bigger.
- **TRUSTED — DO NOT / CANNOT fuse the square chains** (`square_n1024/2048`): fusion is structurally
  **infeasible** (needs 80/144 KiB > the measured 64 KiB SMEM). Most robust result — a hard bound
  from a measured constant.
- **LEAN** — `chain2` (2-GEMM) FUSE ~1.55× (unfused axis validated to 2%).
- **WITHIN ERROR BARS** — `multi_w256` FUSE 1.23× (sign plausible, magnitude ≈ the estimator's error).
- **NOT TRUSTWORTHY** — `multi_w512` "FUSE" 0.2% — a tie inside the error; honest reading: fusion
  doesn't help at w=512 (both sides compute-bound).

## The honest gap

Only the **unfused** chains were measured (a true fused multi-GEMM MACA kernel couldn't be authored
here), so every **fused** time is estimator-predicted. The unfused axis itself under-predicts the
measured chains (geomean est/meas 0.69, worst 0.55–0.60 on narrow chains, best 0.98 on the 2-GEMM
chain), so absolute fused ms are ~1.3–1.8× optimistic — but the *relative* fuse/unfuse verdict is
likely **conservative** (real fusion also eliminates L−1 kernel launches the model gives no credit
for). Lead with the relative verdicts and the structural-infeasibility calls; treat absolute fused
ms as model-only. Validating one real fused kernel (Triton / torch.compile) is the next step.

## Reproduce
```
conda run -n fusion python metax_measure.py     # measure on C500 -> metax_measured.json
conda run -n area   python metax_compare.py      # estimator (C500 model) vs measured
# fusion studies on C500: import metax_c500_model (registers 'metax-c500'), then call
#   chain_gemm_fusion.run(GPUS['metax-c500']) / chain_gemm_focused.run(...) / multi_gemm_fusion.analyze(...,GPUS['metax-c500'])
```
