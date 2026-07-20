# Does SMEM starvation make fuse-all stop being preferred past some depth? (user hypothesis)

**Hypothesis:** "By snowcat, a GEMM's performance grows with the SMEM used per single-tile pass.
The more stages fused, the less SMEM is left for each stage's tile, so past some depth N* fusing
ALL stages should stop being preferred."

Tested by extending the multi-GEMM model with SMEM competition (each fused stage's tile budget =
`SMEM − resident activations`, re-timed by the snowcat estimator via `dataclasses.replace`), swept
L up to 16 across widths, two residency schedules. Adversarially audited (model sound, no bugs).
Code: `multi_gemm_smem.py`. Log: `notes/multi_gemm_smem_plan.md`.

## Verdict: the premise is right in general, but the conclusion is **not** demonstrated — it's a
feasibility artifact, not the graded slowdown hypothesized.

### 1. The premise is true — but it doesn't engage where chain fusion lives
Snowcat perf *does* grow with tile SMEM: for a big square GEMM (8192³), dropping H100 SMEM
227→32 KiB slows it **286µs → 1304µs (4.5×)** as tiles shrink 512×64 → 64×64.

**But the narrow-w GEMMs that make fusion attractive are SMEM-insensitive.** For w ≤ 1024 the
estimator pins the **64×64×32 floor tile with pipeline depth C=2** (because `c_sat=1` — narrow
GEMMs don't demand a deep pipeline) unchanged from 227 KiB **down to ~32 KiB**. Both SMEM→perf
channels (tile width, pipeline depth) are already saturated at the floor, so reducing the budget
is **inert on time** for every feasible segment (verified: C = 2/4/8 give identical roofline
time). The premise, verified on wide GEMMs, does not transfer to the narrow regime.

### 2. No gradual perf crossover exists anywhere (0 of ~160 cells)
Across all (w, schedule, L), there is **no case where fuse-all is feasible but a split is strictly
faster.** Whenever fuse-all is buildable it is strictly optimal (or an exact tie). The graded
"tile-shrink makes a split win" mechanism is never observed — because of the **m0-dodge:** the
fused kernel pins the row-block to the MMA floor (m0=16), which minimizes resident and maximizes
tile SMEM; since narrow weights are L2-resident, the extra row-blocks are free (compute-bound), so
per-stage time is **dead flat** as depth grows (w=512: 0.0695 ms/stage from L=2..11 while tile SMEM
falls 163→19 KiB, C pinned at 2).

### 3. The only crossover is a feasibility cliff — and only under an unphysical schedule

| w | `full` (accumulate-all — **unphysical**) | `seq` (buffer-reuse — **realistic**) |
|---:|---|---|
| 512 | N* = **12** (INFEASIBLE cliff) | none through L=16 |
| 1024 | N* = **6** | none through L=16 |
| 2048 | N* = **3** | none through L=16 |

Under `full`, fuse-all is strictly optimal up to N*−1, then goes **hard infeasible** at
`N* = floor(SMEM/(m0_min·w·bpe)) ≈ floor(6368/w)` — the depth-growing resident activations exceed
SMEM. It's a **residency-formula threshold, not a performance optimum** (sensitive to the
`MIN_TILE_SMEM`/`STREAM_OVERHEAD` constants). And `full` is **physically unrealistic**: a linear
chain only needs its input+output activation live at any stage (~2 adjacent = `seq`); no real
kernel holds *all* boundary activations resident. Under the realistic `seq` schedule the tile
budget is depth-independent and **fuse-all stays optimal at every width through L=16.**

### 4. Why the graded effect is structurally squeezed out
The graded crossover would need a reason to keep m0 *large* (so the kernel can't dodge) — i.e.
**expensive, non-L2 weight re-reads**, which need `w·w·2 > 30 MB` → `w > ~3966`. But fusing even
two stages becomes infeasible at `w ≈ 3184` (seq) / `2120` (full). **The fuseable window
(w ≲ 3184) and the expensive-weight window (w ≳ 3966) are disjoint**, so tile-starvation as a
*performance* effect can never surface here — it only ever appears as a feasibility limit.

## Bottom line — is the expectation reasonable?

**Partly.** The physical intuition (SMEM→perf, and fusion consumes SMEM) is sound, and there *is*
a maximum fusion depth that shrinks as GEMMs widen — matching your direction. **But** in the
snowcat model the effect does **not** show up as the graceful "a shorter fusion becomes faster"
rollover you described:
- For the **narrow GEMMs where chain fusion actually pays off**, the tile/pipeline are saturated at
  the floor, so more fusion costs **zero** tile-efficiency until SMEM simply runs out.
- The crossover that appears is a **hard feasibility cliff** and only under an **unphysical
  accumulate-all schedule**; the realistic buffer-reuse kernel has **no crossover** because it
  dodges starvation with a smaller row-block at no cost.
- A genuine graded crossover needs wide, L2-spilling weights that force a large row-block — but
  that regime is already infeasible to fuse, so the two windows never overlap.

So: reasonable as intuition, but the snowcat model says fuse-all keeps winning (or stays feasible-
optimal) for the shapes where fusion matters. The real ceiling on fusion depth here is **SMEM
capacity (feasibility)**, not a per-tile performance rollover.

## Where the effect WOULD be real (next step, if wanted)
Non-uniform / weight-heavy chains where some segment's weight exceeds eff-L2 (DRAM re-reads),
forcing a large m0 that the SMEM starvation then penalizes — the one regime that couples "want big
m0" with "big m0 starves tiles." That's the genuine test of the graded hypothesis.

## Addendum — square GEMMs (M=N=K=n): clearer, and they explain the whole squeeze

The original chain used extreme tall-skinny GEMMs `[131072, w] @ [w, w]` (only the weight is
square). Re-running with **truly square** GEMMs `[n,n] @ [n,n]` sharpens two things:

**(a) Square GEMMs ARE SMEM-sensitive** (the premise engages, unlike tall-skinny):

| n | time at SMEM 227 / 128 / 64 / 32 KiB | sensitive? |
|---:|---|:--:|
| 512 | 0.6 / 0.6 / 0.6 / 0.6 µs | no (floor tile) |
| 1024 | 2.2 / 2.2 / 2.2 / 2.2 µs | no |
| 2048 | 17.9 / 21.9 / 30.3 / 30.3 µs | **yes (1.7×)** |
| 4096 | 143 / 178 / 336 / 652 µs | **yes (4.5×)** |
| 8192 | 1146 / 1360 / 2640 / 5199 µs | **yes (4.5×)** |

**(b) But square GEMMs can't be fused where it matters** — the fusable window and the
save-worthy window are DISJOINT:

| n | C_i | vs L2 | fuse-all | unfused | fusion helps? |
|---:|---:|:--:|---|---|:--:|
| 1024 | 2 MiB | in L2 | 0.0067 ms | 0.0067 ms | **no — exact tie** (nothing to save) |
| 2048 | 8 MiB | in L2 | 0.0537 ms | 0.0537 ms | **no — exact tie** |
| 4096 | 32 MiB | **spills** | **INFEASIBLE** | 0.4297 ms | can't fuse (held too wide) |
| 8192 | 128 MiB | spills | INFEASIBLE | 0.7162 ms | can't fuse |

For square GEMMs fusion is pointless below n≈2048 (the intermediate stays in L2, so there is no
HBM round-trip to eliminate → fuse-all exactly ties unfused) and infeasible at n≥4096 (the
intermediate spills L2 so a round-trip *would* be worth saving, but a row-block of a width-4096
intermediate won't fit SMEM).

**The underlying law (why this is fundamental):** to be worth fusing, an intermediate must exceed
**eff-L2 = 30 MB ≈ 15M elements** (so there's a DRAM round-trip to avoid); to be fusable, a
row-block slice of it must fit **SMEM = 227 KB ≈ 113K elements**, so the intermediate WIDTH must
be ≲ 3500. A *square* intermediate n×n > L2 needs n > 3872 — but a holdable width needs n < 3500.
**Disjoint.** A *tall-skinny* intermediate M×w > L2 reaches the L2-spill through a huge M while
keeping the width w small and holdable — which is the ONLY shape that satisfies both, and exactly
why real chain fusions (flash-attention, fused MLP) are always batched/tall-skinny. So the SMEM
starvation effect (square, SMEM-sensitive) and the fusion benefit (tall-skinny, SMEM-insensitive)
live in **disjoint shape regimes** — you cannot see both at once.

## Reproduce
```
conda run -n area python multi_gemm_smem.py --w 512 --M 131072 --Lmax 16 --smem seq    # no crossover
conda run -n area python multi_gemm_smem.py --w 1024 --M 131072 --Lmax 8 --smem full   # cliff at N*=6 (artificial)
```
