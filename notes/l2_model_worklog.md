# Work Log — Tier-1 L2 reuse-distance model + ncu calibration harness

Owner: (assisted by Claude Code)  ·  Started: 2026-07-08
Target: `gemm_time_estimator.py` on RTX 4060 Laptop (AD107, sm_89), CUDA 12.8, env `profiling`.

## Goal
Model L2-cache absorption of the snowcat "re-read" traffic so the estimator stops
over-counting DRAM traffic under the "L2 always miss" assumption. Add an Nsight
Compute (ncu) harness to *measure* the real L2 hit rate / DRAM bytes and calibrate
the model's effective-capacity knob `alpha` instead of guessing associativity.

## Design (agreed)
Static mapping ⇒ analytical **reuse-distance** analysis, not cache simulation.
Per operand X ∈ {A=[M×K], W=[K×N], OUT=[M×N]}:
```
dram_bytes(X) = cold(X) · sector_inflation(X) · miss_factor(X)
cold(A)=M·K·bpe, cold(W)=K·N·bpe, cold(OUT)=M·N·bpe
mult(X)         = snowcat_read_bytes(X) / cold(X)            # authoritative re-read count
reuse_dim: A→N, W→M, OUT→K   (each operand omits exactly one dim)
reuse_distance(X) = Σ_Y footprint_Y over loops inner to reuse_dim(X)
                    footprint_Y = bpe · Π_{d∈dims(Y)} (full[d] if d inner else tile[d])
frac_cached(X)  = min(1, C_eff / reuse_distance(X))          # LRU-style partial capture
miss_factor(X)  = 1 + (mult(X) - 1)·(1 - frac_cached(X))     # 1 if mult<=1
sector_inflation(X) = max(1, 32 / (contig_tile_bytes(X)))    # row-major; 32B sector
C_eff = alpha · L2_bytes                                      # alpha absorbs assoc/concurrency
```
- Chip-level L2 capacity ⇒ shared/broadcast operands (small A read by all N-tiles)
  cache naturally; `alpha` folds associativity + concurrency + slice non-uniformity.
- Ada facts used: 128 B line = 4×32 B **sectors** (traffic counted in 32 B sectors);
  L2 physically sliced per memory controller (non-uniform) → not modeled explicitly,
  absorbed into `alpha`. Assoc/replacement undocumented for Ada → intentionally not simulated.
- `--no-l2` recovers exact current behavior. `--pin a|w` models Ada L2 persistence
  (cudaAccessPolicyWindow) by forcing miss_factor=1 for that operand.

## Plan / task list
- [x] 1. Work log + task tracking (this file).
- [x] 2. Estimator: GpuModel L2 fields; reuse-distance helper; sector model;
        concurrency (shared/private) factor; traffic integration; CLI
        (`--no-l2`, `--l2-alpha`, `--pin`); Estimate fields; reporting.
- [x] 3. Test estimator: compiles; `--no-l2` == pre-change; L2-on on 3 decode GEMMs;
        sector edge case; pin flag. (All pass — see results below.)
- [x] 4. ncu harness `l2_calibrate.py`: run ncu on the CUTLASS `gemm` binary for a
        shape+config, parse `dram__bytes_read.sum`, `lts__t_sector_hit_rate.pct`,
        `lts__t_sectors.sum`; compare to model; fit `alpha`. DONE.
- [x] 5. Test harness: ncu present (2025.1.1) but counters BLOCKED on this box; built
        --print-cmd / --from-csv fallbacks; validated parse+fit on a synthetic CSV.
- [x] 6. Final: log updated (below); summary to user.

## ncu harness (step 4-5) — `l2_calibrate.py`
Three modes so calibration survives the counter-permission block:
- `--print-cmd`  : emit the exact `ncu` command (prefix `sudo` / add `--log-file`).
- `--from-csv F` : parse a saved `ncu --csv` capture, fit alpha offline (no GPU).
- live (default) : run ncu, parse, fit; clean ERR_NVGPUCTRPERM guidance on failure.
Metrics: `dram__bytes_read.sum`, `dram__bytes_write.sum`, `lts__t_sectors.sum`,
`lts__t_sector_hit_rate.pct`, `gpu__time_duration.sum`. Picks the main GEMM kernel
by duration (or `--kernel-regex`). Fits alpha by sweeping and matching measured DRAM
total; cross-checks model-implied vs measured L2 hit rate; appends to
`calibration_runs.csv` (pool shapes -> median alpha).

### Counter-permission status on THIS box (WSL2, RTX 4060 Laptop)
- `ncu` = /home/shuhan/miniconda3/envs/profiling/bin/ncu, v2025.1.1 — present.
- Counter collection returns **ERR_NVGPUCTRPERM** (non-admin). No passwordless sudo.
- => Cannot capture on this session. To calibrate, the USER runs (in their shell):
  ```
  python l2_calibrate.py ... --print-cmd            # get the command
  sudo ncu ... --log-file run.csv ./gemm ...         # capture (admin)
  python l2_calibrate.py ... --from-csv run.csv       # fit alpha, log row
  ```
  (In Claude Code the user can run the sudo step via `! sudo ncu ...`.)

### Harness validation (synthetic CSV, since counters blocked)
Fed a hand-made ncu CSV (2 kernels: self-test + cutlass). Harness correctly picked
the cutlass kernel by duration, parsed 65 MB read + 10 MB write (unit-scaled),
71.5 MiB total, hit 54.3%; fit ran and logged. Confirms parse + unit scaling + kernel
selection + alpha sweep + CSV logging all work. (Synthetic row then deleted.)

## Open items / next calibration
- Real alpha is UNKNOWN until the user captures ncu on a few shapes (up_gate, down,
  router, a square). Default alpha=0.6 is a placeholder.
- Joint identifiability: matching DRAM bytes (alpha) and matching kernel time also
  needs a BW-efficiency factor (achieved/peak). router's residual (est 0.41x meas)
  is BW-efficiency, not L2 — confirm by measuring dram__bytes vs the model at the
  fitted alpha (if bytes match but time doesn't, it's BW-eff).
- Concurrency raster assumption (N-fastest) could be refined to match CUTLASS's
  threadblock swizzle if calibration shows systematic bias.

## Files touched
- `gemm_time_estimator.py` — Tier-1 L2 model (+ earlier: latency, split-K, occupancy).
- `l2_calibrate.py` — NEW, ncu calibration harness.
- `notes/l2_model_worklog.md` — this log.

## STATUS: DONE (model + harness implemented & tested; awaiting user ncu capture for real alpha).

## Implementation notes (step 2)
Added to `gemm_time_estimator.py`:
- `GpuModel.l2_bytes` (33554432 = 32 MiB, queried) + `l2_capacity_alpha` (0.6 default).
- `SECTOR_BYTES=32`; helpers `_reuse_distance_bytes`, `_sector_inflation`,
  `_l2_concurrency`.
- Per-operand traffic: `dram = cold · miss_factor · sector`, with
  `frac_cached = min(1, C_eff / (reuse_distance · concurrency))`, `C_eff = alpha·L2`.
- **Concurrency (shared/private)**: chip-shared L2 holds every concurrent tile's
  working set; a private operand has `conc` distinct copies competing, a broadcast
  operand `conc=1`. Row-major raster (N fastest): `P=min(mt·nt,num_sm)`,
  A→m_conc=ceil(P/nt), W→n_conc=min(P,nt). (Assumption: this raster; calibratable.)
- CLI `--no-l2` (exact legacy), `--l2-alpha`, `--pin A|W|OUT` (persistence).
- `Estimate.l2_enabled/l2_capacity_eff_bytes/l2_breakdown`; new report block.

## Test results (step 3) — CUTLASS tiles, order MNK
| GEMM | L2-miss T | L2-on T | est time (L2+occ) | CUTLASS meas | est/meas |
|---|---|---|---|---|---|
| up_gate 128×4096×6144 | 97.0 MiB | 50.5 MiB | 0.310 ms | 0.377 ms | 0.82× |
| down 128×6144×2048 | 97.5 MiB | 49.2 MiB | 0.202 ms | 0.162 ms | 1.25× |
| router 4096×256×6144 | 290.0 MiB | 67.1 MiB | 0.309 ms | 0.748 ms | 0.41× |

- `--no-l2 --no-occupancy-bw` reproduces legacy exactly (up_gate 97.0 MiB / 0.3973 ms). ✓
- Sector edge: BK=8 (16 B < 32 B) → A sector inflation 2.00× (3.00 MiB). ✓
- `--pin W` forces W cached (frac 100%) + NOTE. ✓
- Shared/private working: up_gate A conc=1 (broadcast, fully cached);
  router A conc=12 (private → 71% cached); down large 24 MiB weight not cached.
- **Residual for router (0.41×) is BW-efficiency, NOT L2** — skinny N=256 achieves
  ~30-40% of peak BW. α and BW-eff are unidentifiable from wall-clock alone → this is
  exactly what the ncu harness (step 4) resolves by measuring real DRAM bytes + L2 hit%.

## Progress log
- 2026-07-08: Created work log; design captured.
- 2026-07-08: Step 2 done — Tier-1 L2 model (reuse-distance + sector + concurrency)
  implemented. Step 3 done — full test battery passes (table above). Starting step 4
  (ncu harness).
