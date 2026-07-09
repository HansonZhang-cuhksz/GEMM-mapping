# Single-GEMM Latency-Aware Snowcat-Roofline Estimator

Goal: estimate the execution time of **one** GEMM on a **real GPU** (RTX 4060 Laptop
first), given the GEMM size and a **fully specified mapping** (tile size, software
pipelining stages `C`, loop order). No GEMM is run on the GPU — pure analytical model.

This differs from `area.py` / `area_latency.py` / `ffn_*_area*.py`, which *sweep* a
hypothetical chip's area split (SMEM vs. tensor cores) and *search* for the best mapping.
Here the hardware is fixed (real GPU spec sheet + queried device properties) and the
mapping is an input, not a search variable. Same two tools, same roofline concept.

## The two tools reused verbatim

1. **Snowcat / Orojenesis traffic model** (`snowcat_demo.model.traffic`):
   for a given tiling `(BM=m0, BK=k0, BN=n0, loop_order)` it returns
   - `W = buffer_bytes` = one-stage SMEM working set = `A_tile + W_tile + B_tile`
     `= (m0·k0 + k0·n0 + m0·n0) · bytes_per_element`
   - `T = total_bytes` = min HBM backing-store traffic (A reads + W reads + B r/w),
     accounting for tile reuse implied by the loop order.
2. **Latency-aware roofline** (`notes/latency_pipeline_model.md`):
   ```
   ops         = 2·M·N·K
   latency     = HBM_LATENCY_CYCLES / CLOCK_HZ
   inflight    = num_sm · C · W          # chip-level bytes in flight (Little's law)
   BW_eff      = min(BW_physical, inflight / latency)
   compute_time= ops / peak_tensor_flops
   memory_time = T / BW_eff
   time        = max(compute_time, memory_time)
   ```
   Roofline form: `peak = min(compute_roof, OI·BW_eff)`, `OI = ops/T`, `time = ops/peak`.

### Why `num_sm` multiplies the in-flight term

The notes' area study uses `num_sm = 1` (designs a single big SM and scales `bw` down
by `num_sm`). A real GPU runs the tiles across all SMs concurrently. Each SM keeps `C`
pipeline stages (each `W` bytes) outstanding, so chip-level in-flight bytes are
`num_sm · C · W`, and the achievable bandwidth (capped at the physical `BW`) is that over
the memory latency. Equivalent to `num_sm · min(bw_per_sm, C·W/latency)`.

### SMEM feasibility / auto-`C`

Per SM one threadblock holds `C` stages: constraint `C·W ≤ SMEM_per_block`. If the caller
does not pin `C`, we pick the notes' smallest-optimal depth
`C_best = min( floor(SMEM_per_block / W), ceil(BW·latency / (num_sm·W)) )` (≥1). For large
decode GEMMs the latency is trivially hidden, so `C_best` is usually 1.

## RTX 4060 Laptop GPU model (AD107, Ada, CC 8.9)

Device properties queried live via `torch.cuda.get_device_properties(0)`:

| field | value | source |
|---|---|---|
| SMs (`multi_processor_count`) | 24 | torch |
| tensor cores | 96 (4/SM, 4th-gen) | CC 8.9 → 4 TC/SM |
| SMEM per SM | 102400 B (100 KiB) | torch `shared_memory_per_multiprocessor` |
| SMEM per block (opt-in max) | 101376 B (99 KiB) | torch `shared_memory_per_block_optin` |
| L2 | 32 MiB | torch |
| max SM clock | 3105 MHz | `nvidia-smi --query-gpu=clocks.max.sm` |

Spec-sheet / derived constants (editable in the program):
- **FP16/BF16 tensor throughput per core** = 512 FLOP/clock/core (dense, FP32 accumulate).
  Derived from A100: 312 TFLOPS / (432 cores · 1.41 GHz) = 512; same per-core rate on Ada.
  Matches the `512·1e9` placeholder in `area.py`/`decode_ffn_area_report.md`.
- **peak_tensor_flops** = 96 · 512 · clock. At 3105 MHz → 152.6 TFLOP/s (theoretical).
  NOTE: a laptop AD107 is TGP-limited and will not sustain 3105 MHz under a tensor load;
  realistic sustained boost is ~1.9–2.4 GHz. `GPU_CLOCK_HZ` is exposed so the user can
  dial in a sustained clock. Roofline uses the theoretical ceiling by default.
- **HBM bandwidth** = 256 GB/s (GDDR6, 128-bit bus, 16 Gbps effective). Editable.
- **HBM latency** = 500 cycles (placeholder, same as the A100-based area study). Editable.
- **bytes_per_element** = 2 (BF16/FP16).

## Diagnostics (wave quantization)

Pure roofline assumes the GEMM saturates the whole chip. For decode FFN, `M` is small
(e.g. 128 per expert), so the output-tile count `(M/BM)·(N/BN)` can be < `num_sm`,
underutilizing SMs. The estimator reports `tiles`, `waves = ceil(tiles/num_sm)`, and
`sm_utilization`, and can optionally scale compute_time by the wave-quantization
inefficiency `waves·num_sm / tiles`. Default headline number is the pure roofline (matches
the notes concept); the wave-adjusted time is reported alongside as a refinement.

## File: `gemm_time_estimator.py` (project root)

Pure Python (no numpy) so it runs as-is in the `profiling` conda env. Imports only
`snowcat_demo.model` for the traffic tool. CLI + importable `estimate_gemm_time(...)`.
`--optimal` picks the snowcat min-traffic mapping; `--demo` runs the decode-FFN set;
`--stages C` pins the pipeline depth (else auto smallest-optimal); `--clock-mhz` overrides
the SM clock for a sustained-boost estimate.

## Validation (`validate_estimator.py`, needs GPU + torch)

Estimate (using the snowcat-optimal mapping) vs. measured FP16 `torch.matmul` on the
actual RTX 4060 Laptop:

| gemm | MxNxK | est ms | meas ms | est/meas |
|---|---|---:|---:|---:|
| router | 4096x256x6144 | 0.598 | 0.75 | 0.80 |
| up_gate | 128x4096x6144 | 0.299 | 0.41 | 0.72 |
| down | 128x6144x2048 | 0.137 | 0.15 | 0.92 |
| square2k | 2048x2048x2048 | 0.819 | 0.86 | 0.96 |
| big | 4096x4096x4096 | 6.42 | 7.02 | 0.92 |

The estimate is a consistent optimistic lower bound (72–96% of measured) — expected for a
roofline that assumes peak BW and ignores launch/tail overhead and <100% BW efficiency.
All decode GEMMs are memory-bound, so the target regime is well captured. Note the compute
roof uses the 3105 MHz theoretical peak; the laptop sustains far less (`big` measures
~19.5 TFLOP/s), but compute-bound accuracy is irrelevant for memory-bound decode. Use
`--clock-mhz` to calibrate compute-bound cases.

## STATUS: DONE

`gemm_time_estimator.py` + `validate_estimator.py` implemented and validated. Runs in
`conda run -n profiling` (pure Python, no numpy needed).
