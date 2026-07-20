# Fusion time estimator — plan & log

Goal: use GEMM-mapping's **latency-aware snowcat-roofline single-GEMM estimator**
(`gemm_time_estimator.py`, validated to 72–96% of measured on a real RTX 4060) to answer
the *real-world* fusion question: on a **fixed real GPU**, is the **fused optimal kernel**
faster than the **unfused optimal kernels** it replaces? This is how an inference framework
actually decides whether a fusion is worth enabling — unlike the area-distribution study
(which sweeps a hypothetical die's SMEM/tensor split).

Scope (confirmed with user): **both `h100-sxm` and `rtx4060-measured`**, workload =
**GLM-5.2 decode, batch 2048** (matches the area study). All six fusions.

## Decode batch-2048 kernel dims (bpe = 2)

BATCH=2048, HIDDEN=6144, INTERMEDIATE=2048, EXPERTS=256, TOP_K=8, TOKENS/EXPERT=64,
N_HEADS=64, V_HEAD_DIM=256.

| GEMM | M | N | K | count | note |
|---|---:|---:|---:|---:|---|
| mla_o | 2048 | 6144 | 16384 | 1 | attention output proj (all batch tokens) |
| router | 2048 | 256 | 6144 | 1 | (context; not fused here) |
| up_gate | 64 | 4096 | 6144 | 256 | per expert; N = 2·INTERMEDIATE (gate+up) |
| down | 64 | 6144 | 2048 | 256 | per expert; K = INTERMEDIATE |

Vector/reduction kernels (memory-bound, traffic-only):
- residual add: 2048·6144 elems, 3·bpe/elem (read y, read x, write sum) = **72 MiB**
- pre-FFN RMSNorm reduction: read 2048·6144·bpe (24 MiB) + tiny stat = **24 MiB**
- SwiGLU activation (aggregate, batch·top_k = 16384 rows): read gate+up + write activated
  = 16384·(4096+2048)·bpe = **192 MiB**

## Model

Reuse `GpuModel` + the roofline + occupancy/wave-quantization + `_auto_num_stages` from
`gemm_time_estimator.py`, **out of place** (import; do not edit the original). Add:

1. **`estimate_vector_kernel(traffic_bytes, gpu)`** — a memory-bound elementwise/reduction
   kernel: many tiles → occupancy ≈ 1, so `time = traffic / bw_peak` (occupancy law with
   active_sm = num_sm). The residual/RMSNorm/SwiGLU epilogues are all low-arithmetic-
   intensity (≤ 8 FLOP/byte ≪ both GPUs' ridge OI), so their **CUDA compute is hidden**
   under memory time — no CUDA-core FLOP roof is needed (documented assumption).

2. **`estimate_fused_gemm(m,n,k, *, count, epilogue, gpu)`** — a GEMM whose epilogue/prologue
   folds a neighbouring op on-chip. Searches tiles for min time (like
   `optimal_mapping_by_time`) but with the fusion's modifications:
   - `out_bytes_factor` — F4 writes activated (N/2) not raw gate+up (N) → 0.5·OUT.
   - `a_bytes_factor` — F5's down reads the 2×-wide gate+up → 2·A.
   - `extra_hbm_per_out_tile` / `extra_hbm_once` — F1/F2 residual read, γ read, partial-RMS.
   - `aux_smem(m0,n0)` — extra on-chip state (residual tile, RMS accumulator) that reduces
     the SMEM budget available to the GEMM pipeline (the real-GPU analogue of the
     area study's SMEM "starvation"; smaller here since real SMEM/block is 99–227 KiB).
   - `count` — grouped-GEMM multiplier (256 experts): traffic·count, ops·count, tiles·count,
     so occupancy reflects all experts running together (fills the GPU).
   Traffic uses the base snowcat+L2 model (via `estimate_gemm_time`) then applies the deltas.

3. **`estimate_ffn_fused(...)`** (F6) — custom GEMM-GEMM: `out[M,HIDDEN] = down(SwiGLU(
   up_gate(x)))`, intermediate on chip. Per m0-row-block: read x once, write out once, read
   both weight matrices once → `mt = M/m0` blocks re-read weights `mt×`; buffer holds the
   resident activated + out accumulators (SMEM-gated, as in the area F6 model). Enumerate m0.

Per fusion: **unfused time = Σ(optimal GEMM times) + Σ(vector-kernel times)**;
**fused time = optimal fused-kernel time**. Report both GPUs, the speedup, and the verdict
(worth it iff fused < unfused). Both use the estimator's own optimal-mapping search, so it is
"fused optimal kernel vs unfused optimal kernels", exactly the framework question.

## Steps
1. [x] Read GEMM-mapping estimator; confirm GPU + workload with user; gitignore GEMM-mapping.
2. [x] Implement `fusion_time_estimator.py` (vector + fused-GEMM + FFN-fused + the 6 fusions).
3. [x] Run on h100-sxm + rtx4060-measured; sanity-check numbers.
4. [x] Write results report + verdict per fusion -> `fusion_time_results.md`.

## Log / notes
- **Env:** the README's `profiling` conda env does not exist on this machine, and `fusion`
  (py3.10) can't import GEMM-mapping's own `snowcat_demo` (`enum.StrEnum` is py3.11+). Ran in
  **`area` (py3.14)**, which imports it natively. `gemm_time_estimator.py` unchanged
  (imported only) — the work is out of place in the new `fusion_time_estimator.py`.
- **F6 buffer fix:** first model held `m0*(INTERMEDIATE+HIDDEN)` (activated + full out
  accumulator) = 256+ KiB even at m0=16 → crashed "no feasible row-block". The real binding
  constraint is just the resident activated `m0*INTERMEDIATE` (down contracts over K=
  INTERMEDIATE); with a ~16 KiB streaming overhead, m0=16 fits the 4060 (84 KiB) and m0=32
  the H100 (152 KiB). F6 is then *feasible but slow* (weight re-reads), not infeasible.
- **Tile floor:** switched candidate tiles to 64×64×32 (matches
  `optimal_mapping_by_time`) to avoid the estimator's tiny-tile collapse (TODO.md).

## Result (both GPUs, decode batch 2048)
F1–F5 **FUSE** (1.002–1.044× on H100, 1.002–1.026× on 4060). **F6 SKIP** — 0.515× (H100) /
0.259× (4060): the full-FFN fusion must hold the activated intermediate resident, but real
per-block SMEM (99–227 KiB) caps the row-block at m0=16–32 < M=64, forcing 2–4× weight
re-reads. **This reverses the area study** (idealized ~1.3 MiB SMEM made F6 the strongest);
real per-block SMEM is the binding constraint that kills the full-FFN fusion.
