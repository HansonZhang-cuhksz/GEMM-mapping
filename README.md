# GEMM mapping playground (CUTLASS vs cuBLAS)

`gemm.cu` benchmarks a mixed-precision GEMM `C[M,N] = A[M,K] · B[K,N]`
(**fp16 or bf16** in, fp32 accumulate, fp32 out, all row-major) across several
**CUTLASS mappings** and compares each against **cuBLAS**. It is correctness-gated
(no kernel's time is reported unless it first matches the reference) and then
**auto-tunes**: it picks the fastest correct mapping for the given size and prints
its full schedule.

## Build & run (conda env `profiling`, CUDA 12.8, RTX 4060 = sm_89)

```bash
./build.sh                       # compiles gemm.cu (~11 kernels × 2 dtypes, ~3-6 min)
./run.sh                         # all mappings, both dtypes, current default size
./run.sh --dtype bf16            # bf16 only
./run.sh --dtype fp16 --m 4096 --n 4096 --k 4096 --iters 100
./run.sh --config stages         # only mappings whose name contains "stages" (+ cuBLAS)
./run.sh --help
```

`--dtype fp16|bf16|both` selects the input element type (default `both`). fp16 and
bf16 share the same tensor-core MMA shape (16×8×16) and mappings; only the input
element and the cuBLAS data type differ.

CUTLASS is a local shallow clone in `./cutlass` (v3.8.0). If it is missing:
```bash
git clone --depth 1 --branch v3.8.0 https://github.com/NVIDIA/cutlass.git cutlass
```

## How correctness is guaranteed (TDD)

1. **CPU reference** GEMM in fp32 — the independent oracle.
2. At startup, **cuBLAS is checked against the CPU reference** on a small problem
   (`[self-test] ... PASS`); the program aborts if this fails.
3. cuBLAS then produces the reference for the full-size problem, and **every
   CUTLASS mapping is checked against it** (`max_rel < 1e-2`) before being timed.
   A failing kernel prints `FAIL correctness` and is excluded from the summary.

(The comparison metric was separately verified to reject a deliberately-wrong
kernel, so a green run is a real guarantee, not a vacuous one.)

Note: `max_rel` is tiny (often `0` for large square sizes, ~1e-5 for skinny ones).
When cuBLAS and CUTLASS both avoid split-K they accumulate K in the same fp32 order
and match bit-for-bit; when cuBLAS switches to split-K (e.g. skinny M) the reordered
accumulation adds a ~1e-5 relative difference, still far inside the `1e-2` gate.

## Can CUTLASS pick the best mapping for a size, and can I read it? (auto-tune)

There is **no runtime CUTLASS oracle** that hands you the optimal tile for a given
size. CUTLASS's native auto-selection is either:
- **offline** — the `cutlass_profiler` tool sweeps many kernels for your shape and
  reports the best (this is the CUTLASS-sanctioned "find the best mapping" path), or
- **compile-time** — the CUTLASS 3.x `CollectiveBuilder` with `KernelScheduleAuto`
  picks a *mainloop schedule* for the architecture (not size-specific).

So the practical runtime answer is **empirical auto-tuning**: run the candidate
mappings and keep the fastest correct one. This program does exactly that and then
**reads the winner's schedule off its CUTLASS type** (`ThreadblockShape`,
`WarpShape`, `InstructionShape`, `kStages` are public members of
`cutlass::gemm::device::Gemm`), printing an `[auto-tune]` block per dtype, e.g.:

```
[auto-tune] best CUTLASS mapping for 128x4096x4096 (bf16): "tb64x64x32_w32x32_s4"
            ThreadblockShape = 64 x 64 x 32
            WarpShape        = 32 x 32 x 32
            InstructionShape = 16 x 8 x 16
            Stages           = 4   (software-pipeline depth)
            Swizzle log      = 1
            -> 0.229 ms, 18.8 TFLOP/s, 0.94x cuBLAS
```

The chosen mapping is size-dependent: a skinny `M=128` GEMM prefers a small
`64×64×32` tile (big tiles waste rows), whereas a large square GEMM prefers big
tiles. Widen the candidate set by adding `reg.add<...>` lines to `run_all_mappings`.

> If you specifically want a *library heuristic* that, at runtime, returns a ranked
> algorithm **with a readable tile id** for a given size, that is **cuBLASLt**
> (`cublasLtMatmulAlgoGetHeuristic` + `cublasLtMatmulAlgoConfigGetAttribute`), a
> different library from CUTLASS. Ask if you'd like that wired in as a third baseline.

## Inspecting the mapping cuBLAS chose (`./inspect_cublas.sh`)

The classic cuBLAS API is a black box, but `cublasGemmEx` dispatches through
**cuBLASLt**, so setting `CUBLASLT_LOG_LEVEL=5` makes cuBLAS print the algorithm it
picked. `inspect_cublas.sh` runs the binary with only cuBLAS active and pretty-prints it:

```bash
./inspect_cublas.sh --m 128  --n 4096 --k 4096 --dtype fp16
./inspect_cublas.sh --m 4096 --n 4096 --k 4096 --dtype bf16
```

Each line decodes as `tile=MATMUL_TILE_<M>x<N>`,
`stages=MATMUL_STAGES_<tileK>x<numStages>`, plus `numSplitsK` and `ctaSwizzling`.
Two representative results (rows/cols look transposed due to the row-major operand
swap — see the script header):

```
128x4096x4096 : tile 128x128, stages 32x1, numSplitsK=3, ctaSwizzling=1   (split-K!)
4096x4096x4096: tile 128x128, stages 32x1,               ctaSwizzling=1   (no split-K)
```

**Key insight:** cuBLAS enables **split-K** only for the skinny `M=128` shape (few
output tiles ⇒ split K across CTAs to fill the GPU). That is exactly why cuBLAS
beats the *non-split-K* mappings on the skinny shape and only ties on the square one.

### Reproducing cuBLAS's mapping in CUTLASS (split-K → parity)

CUTLASS supports the same thing: `device::Gemm<..., SplitKSerial=true>` with a
runtime `split_k_slices` gives a serial, in-place K reduction — the analog of
cuBLAS's `numSplitsK` + `REDUCTION_SCHEME_INPLACE`. The registry entry
`cublas_match_s2_splitK3` mirrors cuBLAS's inspected mapping (tile 128×128×32,
swizzle on, **split-K=3**; stages=2 because CUTLASS's multistage mainloop can't do
cuBLAS's "1 stage"). Result on the skinny `128×4096×4096` shape (fair isolated
head-to-head, 300 iters, this 35 W-capped 4060):

```
without split-K, best CUTLASS : ~0.28 ms   (~0.74x cuBLAS)   <- the gap
cublas_match_s2_splitK3       : ~0.265 ms                    }  parity
cuBLAS                        : ~0.285 ms                    }  (within run-to-run noise)
```

So **yes** — matching cuBLAS's mapping (crucially, adding split-K) closes the gap
and reaches cuBLAS-similar performance; the remaining difference is within the
power-cap noise. Sweep the split count with the `--config splitK` entries
(`splitK2/3/4`). Note that identical *mapping* is necessary but not always
sufficient: cuBLAS ships hand-tuned SASS, so for other shapes a matched CUTLASS
mapping may land close but not exactly on cuBLAS.

## Can a mapping beat cuBLAS? Profiling the neighborhood (`--fair`)

**Measurement caveat first.** On this 35 W-capped 4060, naively timing every
mapping and then cuBLAS *last* is unfair — the GPU is hottest/most throttled by the
time cuBLAS runs, making CUTLASS look artificially good. Use **`--fair`**: it
interleaves each candidate with cuBLAS (A,B,A,B,…) and reports the **median
`cuBLAS_ms / candidate_ms` over N rounds**, so slow thermal drift cancels.

```bash
./run.sh --dtype fp16 --m 128 --n 4096 --k 4096 --fair --iters 80 --fair-rounds 15
```

Sweeping the neighborhood of cuBLAS's optimum (tile 128×128×32, split-K≈3) on the
skinny `128×4096×4096` shape gives a **stable, reproducible** picture:

| mapping | fair speedup vs cuBLAS |
|---|---|
| no split-K (any tile/stages/swizzle) | **~0.72×** (a cliff) |
| split-K = 2 or 4 | ~0.93–0.94× |
| split-K = 6 / 8 | ~1.00× / ~0.98× |
| **split-K = 3** (several tiles) | **~1.02–1.05×** ✅ |
| best: `tb128x128x32_splitK3` (stages=3) | **~1.03×** (repeatable) |

**Answers:**
- **Yes, you can beat cuBLAS — but only by ~2–3%**, reproducibly, on this shape/GPU.
  The winners combine cuBLAS's split-K with a *deeper* pipeline (stages=3 vs cuBLAS's
  shallow) or a transposed tile (64×128). Both fp16 and bf16 show the same ~1.03×.
- **cuBLAS sits on a broad, shallow plateau, not a sharp peak.** The entire
  split-K≈3–6 neighborhood is within ±5% of cuBLAS; split-K is the one knob that
  actually matters here (it moves you off the 0.72× cliff). That is the expected
  shape of a well-tuned library's operating point: near-optimal, beatable by a few
  percent with exhaustive local search, not by a landslide.

Don't over-read the 2–3%: it is real and repeatable here, but it is the kind of
margin that can flip on a non-power-capped GPU, a different shape, or a newer cuBLAS.
The robust conclusion is **parity ± a few percent**, with split-K being decisive.

These map onto the same knobs CUTLASS exposes: `tile`↔`ThreadblockShape`,
`stages`↔`Stages`, `ctaSwizzling`↔`ThreadblockSwizzle`, `numSplitsK`↔split-K.

Other ways to inspect: `ncu ./gemm ...` shows the actual launched kernel name (e.g.
`ampere_h16816gemm_128x128_...`, which encodes tile/MMA/stages); `nsys` gives a
kernel trace. Both are installed under the `profiling` env (`ncu`).

## The mapping knobs (what you experiment with)

CUTLASS exposes the GPU schedule as **compile-time template parameters**. This is
the direct analog of Triton's autotuning knobs:

| Knob | CUTLASS parameter | Triton analog |
|------|-------------------|---------------|
| **Tiling** (CTA tile) | `ThreadblockShape = GemmShape<M,N,K>` | `BLOCK_M/N/K` |
| **Tiling** (warp tile) | `WarpShape` | (warp-level split) |
| MMA instruction | `InstructionShape` (16×8×16, fp16 & bf16) | fixed by HW |
| **Software pipelining** | `Stages` | `num_stages` |
| **Loop order** | `ThreadblockSwizzle` (raster order) | grid `program_id` order |
| **Split-K** | `SplitKSerial=true` + `split_k_slices` | `SPLIT_K` |

### On "loop order"
CUTLASS does **not** let you reorder the inner K mainloop like rewriting a nested
loop. What it exposes is the **threadblock rasterization order** — the order CTAs
walk the output-tile grid, via `GemmIdentityThreadblockSwizzle<N>` (N = log2 of
the tile-group width, trading which output tiles run concurrently for L2 reuse).
That rasterization is the practical "loop order" knob (`swizzle_grp*` entries).

## Two ways to experiment

1. **Edit + recompile** the `PRIMARY MAPPING` macro block at the top of `gemm.cu`
   (`CFG_TB_*`, `CFG_WARP_*`, `CFG_STAGES`, `CFG_SWIZZLE`, ...). This drives the
   `custom (macros)` entry. Because these are compile-time template args, changing
   a schedule requires a rebuild.
2. **Pick at runtime** from the pre-built `REGISTRY` of named mappings with
   `--config <substring>` (no recompile). Add your own with one line, e.g.
   `reg.add<GemmMappingT<E, 128,64,32, 64,32,32, 16,8,16, 4, 1>>("my_map", 1);`
   (the trailing arg is the swizzle log, since it can't be read back off the type).

## Caveat on absolute numbers

`nvidia-smi` reported a **35 W** power cap on this laptop 4060, so absolute
TFLOP/s (~20) are power-limited, not representative of the silicon's peak. Since
cuBLAS runs in the same envelope, the **relative** CUTLASS-vs-cuBLAS comparison is
the meaningful output.
