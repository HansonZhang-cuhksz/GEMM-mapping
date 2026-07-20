# Multi-GEMM chain fusion: is deeper fusion better than shorter fusion + splits?

Chain of L GEMMs, **uniform narrow width** (the fusion-preferring regime): `X[M,w] @ W1[w,w] @
… @ WL[w,w] → Y[M,w]`, M=131072 so every intermediate `C_i = M·w` spills the 30 MB eff-L2. A
*partition* cuts the L stages into fused **segments**; each cut materializes that intermediate
to HBM (a round-trip). We enumerate all `2^(L-1)` partitions and time every GEMM with the
snowcat-roofline estimator (`multi_gemm_fusion.py`, generalizing the 2-GEMM `fused_time`).
Verified by a 5-agent workflow incl. an adversarial model audit (model sound; one float
tie-break fixed).

## Answer: fusing MORE stages is **≥** fusing fewer + splitting — strictly better when
memory-bound, an exact tie when compute-bound, never worse (while feasible).

**Fuse-all is the tied-or-strict optimum in every feasible (L, w) cell.** Deeper fusion never
loses; a split can't reduce FLOPs or raise occupancy, so at best it matches fuse-all and at
worst it pays extra HBM round-trips.

### The literal question — 3-GEMM fusion vs 2-GEMM fusion + 1 (memory-bound, w=128)

| partition (L=3) | time | vs fuse-all |
|---|---:|---:|
| **fuse-all-3** | **0.0201 ms** | 1.00× |
| fuse-[1,2] + 3 | 0.0401 ms | 1.99× slower |
| fuse-1 + [2,3] | 0.0401 ms | 1.99× slower |
| fully unfused | 0.0601 ms | 2.99× slower |

Yes — 3-GEMM fusion wins. The two half-fused placements are **identical** because total time
depends on the **number of cuts, not where they fall**. Same ordering at w=256.

### Regime map (L=6, M=131072)

| w | C_i | regime | fuse-all vs unfused | fuse-all optimal? |
|---:|---:|---|---:|:--:|
| 128 | 32 MB | memory-bound | **4.585×** | ✅ strict |
| 256 | 64 MB | memory-bound | 2.303× | ✅ strict |
| 512 | 128 MB | transition | 1.155× | ✅ (mild) |
| 1024 | 256 MB | compute-bound | 1.007× | ✅ exact tie |
| 2048 | 512 MB | compute-bound | 1.015× | ✅ exact tie |
| ≥4096 | ≥1 GB | — | fuse-all **infeasible** | forced unfused |

Time is ~**linear in the number of segments**: at w=128, `total ≈ fuse-all + (#segments−1)·(one
round-trip)`, each cut = **+64 MiB = 2·C_i** = +0.020 ms (exactly 64 MiB / 3.35 TB/s, HBM peak).

## Mechanism

Every cut forces intermediate `C_i` to be written to and re-read from HBM: **+2·C_i, one round-
trip per cut.** Whether it *costs time* is a roofline question:

- **Memory-bound (narrow w):** the fused kernel is already bandwidth-saturated and `C_i > L2`
  can't stay resident, so each round-trip lands on the critical path at HBM peak → more cuts =
  strictly more time, dead-linear in segment count, independent of cut location. **Deeper fusion
  strictly wins.**
- **Compute-bound (wide w):** the round-trip is added only to the *memory* roof, which sits
  5–20× below the *compute* roof (fuse-all compute/memory = 5.2× at w=512, 20× at w=2048), so it
  hides completely under the GEMM math. Compute time is algebraically partition-invariant (FLOPs
  fixed; occupancy `out_tiles = mt·(w/bn)` is set by the M×w output grid, identical for one big
  kernel or several small ones). **Exact tie — depth is free.**
- **Infeasible (w ≥ 4096):** paired-activation residency `peak_pair = 2w` exceeds SMEM (m0_max=13
  < the 16-row MMA min), so no fused segment of length ≥ 2 can be built. **Forced fully unfused.**

## Marginal benefit of each added fused stage

In the memory-bound regime it stays **positive and roughly constant** — never turns over. Each
extra fused stage removes one more `C_i` round-trip, so absolute time saved grows ~linearly in L
(w=128 saves 0.020/0.040/0.060/0.078/0.094 ms as L=2→6). The *relative* benefit shrinks as w
grows (compute ~w², memory saving ~w), which is why depth matters enormously when narrow/memory-
bound and becomes negligible (exact tie) when wide/compute-bound — but it is **never negative in
any feasible regime**, so "fuse all" dominates "fuse fewer + split" throughout.

## Scope caveats (from the audit)

- **This is the uniform-narrow answer.** For uniform widths the fused weight re-read cost
  (`w·w·2 > L2`) is *inert* for every feasible segment (it only turns on at w ≥ 4096, which is
  always infeasible), so fusion carries ~zero modeled cost beyond the saved round-trips — making
  fuse-all optimal *partly by construction*. A genuine "fuse-fewer-wins" crossover requires
  **non-uniform / large weights** (some segment's weight > L2 so its `mt×` re-read goes to DRAM).
  That is the natural next experiment.
- Residency uses `peak_pair = 2w` (hold two adjacent activations) — more conservative than the
  2-GEMM `min(N1,N2)`; it only sets the feasibility cap, feasible timings unchanged.
- Assumes `C_i > L2` (holds for M=131072, w ≥ 128); a warning fires if violated (small M·w),
  where cuts would instead stay L2-resident.

## Reproduce
```
conda run -n area python multi_gemm_fusion.py --L 6 --w 128 --M 131072 --verbose   # fuse-all optimal 4.585x
conda run -n area python multi_gemm_fusion.py --L 6 --w 1024 --M 131072            # compute-bound: exact tie
```
