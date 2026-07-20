# Multi-GEMM fusion + SMEM-starvation (user hypothesis) — plan & log

## User hypothesis
"By snowcat, GEMM perf grows with the SMEM used per single-tile pass. The more stages fused, the
less SMEM is left for any single stage's GEMM tile, so perf should drop past some fusion depth."
Task: run for >3 GEMMs, find at how many stages fuse-all stops being preferred; assess if reasonable.

## Step 1 — verify the premise in the estimator  [done]
`_auto_num_stages` (gemm_time_estimator.py:419): `c_max = smem_per_block // w`, pipeline depth
`c_best = min(c_max, max(c_sat,1))`, and `bw_latency = active_sm*C*W/latency`. So SMEM caps the
pipeline depth C; if `c_max < c_sat` the GEMM can't saturate HBM -> slower. Also tile selection
(`best_at_capacity`) picks a smaller BMxBN when SMEM is small. EMPIRICAL (H100, reduce
smem_per_block):
 - big square 8192x4096x4096: 227KiB->286us, 96KiB->663us, 32KiB->1304us (4.5x). SMEM-SENSITIVE.
 - narrow 131072x128x128 and m0=256,w=128: FLAT 0.159/20us from 227 down to 48KiB, then a cliff
   (infeasible < 24KiB). Narrow GEMMs are NOT SMEM-sensitive (small tiles already optimal), only
   an infeasibility cliff.
=> Premise TRUE, but strength is width-dependent: strong for wide (compute-capable) GEMMs, weak
   (cliff-only) for narrow ones.

## Step 2 — the depth dependence (key subtlety)
The per-stage SMEM penalty creates a DEPTH crossover only if the resident state GROWS with the
number of fused stages. Two schedules:
 - SEQ / buffer-reuse (tiny-cuda-nn style fused MLP): reuse one activation buffer across layers ->
   resident ~ m0*2w, DEPTH-INDEPENDENT -> a fixed per-stage SMEM penalty, NO depth threshold.
 - FULL / accumulate: hold all boundary activations resident -> resident ~ m0*(k+1)*w, GROWS with
   depth k -> tile SMEM shrinks with depth -> perf drops past some N* -> the user's crossover.
Model both (module `multi_gemm_smem.py`), reduce each fused stage's tile-SMEM budget to
`SMEM - resident` via dataclasses.replace, and sweep L to locate N*(w, schedule).

## Step 3 — sweep L, find N*  [done, memoized segment_time by (widths,M,schedule)]

N* = smallest L at which fuse-all is NOT optimal (M=131072, H100):
| w | FULL (accumulate) N* | SEQ (buffer-reuse) N* |
|---|---|---|
| 512  | 12 (INFEAS cliff) | none through L=12 |
| 1024 | 6  (INFEAS cliff) | none through L=12 |
| 2048 | 3  (INFEAS cliff) | none through L=12 |

Key observations:
 - FULL: fuse-all is STRICTLY optimal right up to N*-1, then INFEASIBLE at N* (resident exceeds
   SMEM). It is a FEASIBILITY CLIFF, not a gradual perf crossover. N* ~ SMEM/(m0_min*w*bpe),
   so smaller for wider w (12 @512, 6 @1024, 3 @2048). Matches user's direction.
 - SEQ (depth-independent resident): NO crossover at all — fuse-all stays optimal. The kernel
   DODGES tile-SMEM starvation by shrinking the row-block m0 (costless because uniform weights are
   L2-resident, so more blocks = free re-reads). So the snowcat starvation never bites on perf.
 - Whenever fuse-all is FEASIBLE it is strictly optimal — there is never a "fuse-all feasible but a
   split is faster" gradual crossover. A gradual perf crossover would need expensive (non-L2)
   weight re-reads to force a large m0, but that needs wide w -> resident cap makes it infeasible
   first. The effect is squeezed out.

Verdict shaping: premise TRUE; but "fuse-all stops being preferred past N*" only materializes as a
FEASIBILITY limit under an accumulate-all (SMEM-wasteful) schedule; under the efficient buffer-reuse
schedule an optimizing kernel sidesteps starvation with a smaller row-block, so no crossover.

## Step 4 — workflow (4 agents: 2 experiments + adversarial audit + synth)  [done]

Audit verdict: model SOUND (no correctness bugs; feasibility/tie-break/memo all verified), but the
graded hypothesis is NOT demonstrated. Two high-severity interpretation caveats:
 1. 'full' residency is UNPHYSICAL and is the SOLE source of the crossover. A linear chain needs
    only ~2 adjacent activations live (=seq) — no real kernel holds all boundary activations. So the
    seq/full CHOICE, not physics, decides whether a crossover exists; the honest bracket is ~seq at
    both endpoints -> NO crossover.
 2. Narrow-w tiles are SMEM-INSENSITIVE in the estimator: 64x64x32 floor tile held from 227->32 KiB,
    C pinned at MIN_NUM_STAGES=2 (c_sat=1, c_best_auto=1). So both SMEM->perf channels (tile width
    bn, pipeline depth C) are saturated at the floor -> reduced SMEM is INERT on time; graded
    starvation is structurally impossible here. N* = floor(6368/w) is a residency-formula threshold
    set by MIN_TILE_SMEM(12KiB)+STREAM_OVERHEAD(16KiB)+resident(m0=16), not a perf optimum.

Confirmed: 0 of ~160 (w,schedule,L) cells have fuse-all feasible-but-slower-than-a-split. The graded
crossover would need expensive non-L2 weights (w>~3966) to reward large m0, but fusion dies at
w~3184(seq)/2120(full) -> the fuseable and expensive-weight windows are DISJOINT -> squeezed out.

Actions taken: default schedule -> 'seq' (realistic); docstring reframes 'full' as artificial upper
bound + documents narrow-w saturation. Verdict written to multi_gemm_smem_results.md.
