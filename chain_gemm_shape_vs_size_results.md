# Is 2-GEMM chain fusion preference more about SHAPE or SIZE?

Five experiments against the same snowcat-roofline probe (`chain_gemm_probe.probe`), each
adversarially verified, then synthesized. Scripts: `expA_scale_invariance.py`,
`exp_B_shape_at_fixed_size.py`, `exp_c_ratio_collapse.py`, `exp_d_importance.py`,
`exp_E_hardware_causal.py`.

Definitions — **SHAPE** = aspect ratios (scale-free: `aA=log2(M/K1)`, `aB=log2(K1/N1)`,
`aD=log2(N1/N2)`). **SIZE** = absolute magnitude (geomean dim, FLOPs). **RATIOS** = size-vs-
hardware (`f_B=K1·N1·2/L2`, `f_D=N1·N2·2/L2`, `f_C=M·N1·2/L2`, `f_held=min(N1,N2)·2/SMEMblk`,
`AI_fused/ridge`).

## Verdict: it's the size-vs-hardware **RATIOS**; of the two raw primitives, **SIZE ≫ SHAPE**

Neither pure shape nor pure absolute size decides it alone. The verdict collapses onto
dimensionless ratios — *"size measured in units of L2 and SMEM capacity"* — and **shape only
selects which ratio binds first.** If forced to pick one raw primitive, the honest answer is
**SIZE**: pure aspect ratios can't even beat the majority baseline.

| Experiment | What it isolates | Result |
|---|---|---|
| **A — scale-invariance** | fix shape, scale size | Verdict is **NOT scale-invariant**: scaling 6 fixed shapes flips **5 of 6** (a shape-only rule predicts 0 flips). → **size is decisive; shape alone cannot determine the verdict.** |
| **B — fixed-size shape sweep** | fix size, vary shape | At fixed size, shape **decisively** changes the verdict (entropy 0.93–0.99 bits; at geomean 4096: 10 FUSE / 67 unfuse / 8 infeasible — all three coexist). But because the thresholds are on **pairwise products**, shape earns influence only by moving those products across the fixed L2/SMEM lines. |
| **C — ratio collapse** | do ratios explain it? | Depth-3 tree: **ratios 0.913** vs shape 0.777 vs size 0.803 (base 0.75). A single stump `AI/ridge>0.55` scores 0.870 — beating the entire shape and size trees. Feasibility = f(`f_held`) at **acc 1.000**. |
| **D — feature importance** | rank the groups | Importance mass **RATIO 0.586 > SIZE 0.314 ≫ SHAPE 0.100**; CV acc ratio 0.905 / size 0.806 / **shape 0.711 (below the 0.727 baseline)**. Ratio-only classifier reproduces the full model. |
| **E — hardware causal** | is it size-vs-capacity? | Doubling L2 + SMEM **translates every boundary to larger dims for the same shape** → raw-dims-as-lever and shape-as-lever both **falsified**. Flips land as `f_B`/`f_D` cross 1; feasibility onset at a fixed `f_held`. |

## Mechanism

The estimator thresholds the fuse decision directly on fixed capacities (~30 MB eff-L2, per-
block SMEM), so the true levers are ratios of problem quantities to those capacities:

- **`f_held = min(N1,N2)·2 / SMEMblk`** — feasibility (can a row-block be held?). Infeasible iff
  it exceeds the cap; predicts feasibility at 100%.
- **`f_B, f_D`** — streamed-weight L2 residency → the `mt×` re-read is free (L2) or DRAM.
- **`f_C = M·N1·2 / L2`** — whether the intermediate spills, i.e. whether fusion *has a C
  round-trip to save*.
- **`AI_fused/ridge`** — compute- vs memory-bound (the single most predictive feature).

**FUSE wins in the narrow corner** where the fused kernel is compute-bound (`AI/ridge > ~0.4`)
**and** streamed weights stay L2-resident (`f_B, f_D < 1`) **and** C still spills (`f_C > 1`)
**and** `min(N1,N2)` is feasible. Crucially these thresholds sit on **pairwise products, not
total size** — which is exactly why redistributing exponents at fixed geomean (shape) can push
one product across its line while another stays put. That is the whole reason shape appears to
matter even though size is the lever.

## Regimes as size grows (fixed narrow shape)

1. **SMALL** (all products < L2): no re-read penalty, little C saving → FUSE≈unfuse ties (~1.0×).
2. **TRANSITION** (products cross the L2/SMEM lines): shape matters most; FUSE needs the rare
   corner — tall M keeps `f_C>1`, narrow N1 keeps `f_B,f_D<1` and `min(N1,N2)` feasible.
3. **LARGE** (weights spill L2, `f_B>1`): `mt×` DRAM re-read dominates → unfuse wins broadly.
4. **INFEASIBLE** (`min(N1,N2)` past the SMEM cap): fusion structurally impossible.
5. **NON-MONOTONIC** (tall-A 16:1:1:1): unfuse → FUSE (`f_C` crosses 1) → unfuse (`f_B` crosses
   1) — direct proof that raw magnitude is neither uniformly better nor worse; only your
   position between the thresholds decides.

## Honest caveats (from the adversarial verifiers)

- **Within-model, train-accuracy result.** `f_B/f_D/f_C/f_held/AI-ridge` are the estimator's
  own internal decision variables, so "collapses onto ratios" and "feasibility = f(f_held) =
  1.000" are partly true *by construction*. This characterizes the model, not measured silicon.
- **A (corrected):** the flips are all size-driven (the thesis), but only 3–4 of 5 land exactly
  on a `ratio=1` crossing (the weight-residency ones); flash-attn and square flip through
  smoother compute-bound near-ties, not sharp crossings.
- **E (corrected):** the "every boundary moves *exactly* 2×" headline is dropped — that came
  from linear single-lever sweeps. Under genuine fixed-shape scaling the L2-flip boundary moves
  ~√2 per dim (the weight product scales as size²); only the SMEM feasibility edge is ~2×.

## Bottom line

Fuse preference is **not a shape property and not a raw-size property — it is a size-relative-
to-hardware property.** Between the two raw axes, **size dominates shape** (shape alone is
below chance). Shape's real job is to decide *which* capacity ratio you hit first, because the
binding thresholds live on pairwise products (`K1·N1`, `N1·N2`, `M·N1`, `min(N1,N2)`) rather
than on overall scale.
