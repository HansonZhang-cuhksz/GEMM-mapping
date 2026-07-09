# GEMM Execution-Time Estimator — Comprehensive Reference

**Audience:** an engineer/agent who will maintain or extend `gemm_time_estimator.py`
with access to **only** this document and that one source file. This document therefore
also fully specifies the external `snowcat_demo.model` package that the program imports,
so you do not need to read its source.

---

## 1. What the program does

`gemm_time_estimator.py` predicts the **wall-clock execution time of a single GEMM**
(matrix multiply `C[M,N] = A[M,K] · B[K,N]`) on a **fixed real GPU** (default: NVIDIA
RTX 4060 Laptop), given:

- the **GEMM size** `(M, N, K)`;
- a **mapping** = tile sizes `(BM, BN, BK)`, software-pipelining depth `C` (`num_stages`),
  and tile **loop order** (a permutation of `M,K,N`);
- a **GPU model** (hardware constants).

It is **purely analytical** — no GEMM is executed to produce the estimate. It combines two
tools:

1. the **Snowcat / Orojenesis traffic model** — computes, for the given mapping, the
   on-chip SMEM working-set size `W` and the off-chip HBM traffic `T`;
2. a **latency-aware roofline** — turns `(ops, W, T)` + GPU specs into a time.

### Context (why this exists)

This is part of a die-area / dataflow study for a GPU aimed at **MoE-LLM decode FFN**
(small batch/`M`, weight-streaming-dominated GEMMs). Sibling scripts in the same project
(`area.py`, `area_latency.py`, `ffn_*_area*.py`, `multi_gemm_area.py`) *sweep* a
hypothetical chip's silicon-area split (SMEM vs. tensor cores) and *search* for the best
mapping. **This program is different:** the hardware is a fixed, real GPU and the mapping
is an *input* (not searched). Same two tools, same roofline concept, different question:
"how long does *this* GEMM with *this* mapping take on *this* GPU?"

### What it does NOT model

- Kernel launch overhead, driver/CPU overhead, cache (L2) effects beyond the analytic
  traffic model, imperfect compute/memory overlap, DVFS/thermal throttling.
- Achieved bandwidth < peak, achieved clock < boost. The estimate is an **optimistic
  lower bound** (measured times are ~5–40 % higher; see §7).
- Numerics/accuracy, split-K, stream-K, or any specific library's kernel selection.

---

## 2. Method 1 — Snowcat / Orojenesis traffic model

This model answers: *for a chosen tiling, how many bytes cross the HBM boundary, and how
much SMEM does one tile step need?* It assumes a **single-entry buffer per operand**
(one A-tile, one B/output-tile, one W-tile resident) and counts reloads implied by the
loop order. It is an analytic idealization of a tiled GEMM's data movement.

### 2.1 Definitions and notation

- `M, N, K` — GEMM dimensions. `A` is `M×K`, `B`(weights) is `K×N`, output `Cout` is `M×N`.
  (In this codebase the second operand / output are named `W`/`B` respectively — see the
  naming caution in §2.6.)
- `bytes_per_element` (`bpe`) — 2 for FP16/BF16 (the default and the decode workload).
- Tile sizes `m0=BM`, `k0=BK`, `n0=BN`. **Tiles must exactly divide** their dimension
  (`M % BM == 0`, etc.).
- Tile counts: `mt = M/BM`, `kt = K/BK`, `nt = N/BN`.
- Tile byte sizes:
  - `a_tile = BM·BK·bpe`
  - `w_tile = BK·BN·bpe`
  - `b_tile = BM·BN·bpe`   (output tile)
- **Working set** `W = buffer_bytes = a_tile + w_tile + b_tile` (bytes of SMEM to hold one
  A-tile + one W-tile + one output-tile — i.e. one pipeline stage).
- **Loop order** — an ordered triple, e.g. `("M","K","N")`, listing the tile loops
  **outermost first** (leftmost = slowest-varying). There are 6 permutations (see §2.3).

### 2.2 Operation count

`ops = 2·M·N·K` FLOPs (one multiply + one add per MAC). Independent of the mapping.

### 2.3 The six loop orders

```
LOOP_ORDERS = (("M","K","N"), ("M","N","K"), ("K","M","N"),
               ("K","N","M"), ("N","M","K"), ("N","K","M"))
```

### 2.4 Traffic closed form (the exact algorithm)

Given `(m0,k0,n0,loop_order)`, the model returns a `TrafficBreakdown` with fields
`buffer_bytes, a_read_bytes, w_read_bytes, b_read_bytes, b_write_bytes`, and
`total_bytes = a_read_bytes + w_read_bytes + b_read_bytes + b_write_bytes`.

Let `extents = {"M": mt, "K": kt, "N": nt}` and `pos(dim) = loop_order.index(dim)`
(0 = outermost). The core helper counts how many times a tile keyed on a set of dims is
(re)loaded, given a single-entry cache:

```
run_count(key_dims):
    varying = [pos(d) for d in key_dims if extents[d] > 1]
    if not varying: return 1
    deepest = max(varying)                      # innermost loop that changes this tile
    return product( extents[d] for d in loop_order[: deepest+1] )
```

Intuition: a tile is reused across any loops **inner** to the deepest loop that changes
its key; it must be reloaded once per iteration of everything from the outermost loop down
to that deepest key loop. `run_count` = number of those (outer-prefix) iterations.

Reads for the two input operands:

```
a_read_bytes = run_count(("M","K")) · a_tile        # A keyed on (M,K)
w_read_bytes = run_count(("K","N")) · w_tile        # W keyed on (K,N)
```

Output tile `B` keyed on `(M,N)`. Whether the K-reduction happens **inside** the output
tile's residency decides if partial sums are accumulated in place (write once) or spilled
and reloaded every K step:

```
out_positions = [pos(d) for d in ("M","N") if extents[d] > 1]
deepest_output = max(out_positions, default=-1)
k_inside_output = pos("K") > deepest_output          # K loop is inner to both M and N loops
output_tiles = mt · nt

if k_inside_output:                # output-stationary: accumulate fully, then write once
    b_read_bytes  = 0
    b_write_bytes = output_tiles · b_tile
else:                              # K outside: re-read/re-write partial sums each K step
    b_read_bytes  = output_tiles · max(kt-1, 0) · b_tile
    b_write_bytes = output_tiles · kt · b_tile
```

`buffer_bytes = a_tile + w_tile + b_tile` (independent of loop order).

### 2.5 Worked example (verify your understanding)

`up_gate` GEMM `M=128, N=4096, K=6144`, `bpe=2`, mapping `BM=128, BN=256, BK=1`,
loop_order `("M","N","K")`:

- `mt=1, nt=16, kt=6144`.
- `a_tile=128·1·2=256`, `w_tile=1·256·2=512`, `b_tile=128·256·2=65536`.
- `W = 256+512+65536 = 66304 B = 64.75 KiB`.
- `pos: M=0, N=1, K=2`.
- `a_read = run_count({M,K})`: varying = {K@2} → deepest=2 → `1·16·6144=98304` loads ·256 = 24 MiB.
- `w_read = run_count({K,N})`: varying = {N@1, K@2} → deepest=2 → 98304 ·512 = 48 MiB.
- output: out_positions={N@1}, deepest_output=1; `pos(K)=2 > 1` ⇒ k_inside_output=True ⇒
  `b_read=0`, `b_write = 16·65536 = 1 MiB`.
- `T = 24+48+0+1 = 73 MiB`, `OI = ops/T = 2·128·4096·6144 / 76546048 = 84.16 FLOP/byte`.

This matches the program's `--optimal` output for `up_gate`.

### 2.6 The `snowcat_demo.model` API used by the program (full spec)

The program imports four things. Their exact behavior:

**`GemmWorkload(m, k, n, bytes_per_element)`** — frozen dataclass. **Note the constructor
argument order is `(m, k, n, ...)` — K is second.** Properties:
- `.operations = 2·m·k·n`
- `.a_bytes = m·k·bpe`, `.w_bytes = k·n·bpe`, `.b_bytes = m·n·bpe`
- `.algorithmic_minimum_bytes = a_bytes + w_bytes + b_bytes`
- `.tile_counts(m0,k0,n0) -> (m//m0, k//k0, n//n0)`; raises `ValueError` if not evenly
  divisible.
- `.tile_bytes(m0,k0,n0) -> (a_tile, w_tile, b_tile)` as in §2.1.
- Constructor raises `ValueError` if any of m,k,n,bpe ≤ 0.

**`divisors(n) -> list[int]`** — all positive divisors of `n`, ascending. Raises if `n≤0`.

**`estimate_mapping_traffic(workload, m0, k0, n0, loop_order) -> TrafficBreakdown`** — the
closed form of §2.4. **Argument order is `(m0, k0, n0)` = `(BM, BK, BN)`.** The program
calls it as `estimate_mapping_traffic(workload, mapping.bm, mapping.bk, mapping.bn,
mapping.loop_order)` — i.e. it deliberately passes `bk` in the 3rd slot and `bn` in the
4th. Getting this order wrong silently produces wrong traffic. `TrafficBreakdown` exposes
`.buffer_bytes` and `.total_bytes` (and the per-operand breakdown fields).

**`enumerate_mappings(workload) -> list[MappingPoint]`** — Cartesian product over
`divisors(m) × divisors(k) × divisors(n) × LOOP_ORDERS`; one `MappingPoint` each. Cost
grows with the divisor counts × 6; fine for typical LLM dims (seconds), but can be large
for highly composite dimensions. Each `MappingPoint` exposes:
- `.mapping` — a `GemmMapping(m0, k0, n0, loop_order)` with `.m0/.k0/.n0/.loop_order`.
- `.buffer_bytes` (= `W`) and `.backing_store_bytes` (= `T`, the total HBM traffic).

**`best_at_capacity(points, capacity_bytes) -> MappingPoint | None`** — among points with
`buffer_bytes ≤ capacity_bytes`, returns the one minimizing
`(backing_store_bytes, buffer_bytes)`. `None` if none fit. This is how the program's
`--optimal` selects the **minimum-traffic mapping that fits one threadblock's SMEM**.

(Not imported but conceptually relevant: the "ski slope" / Pareto frontier is
`best_at_capacity` swept over increasing capacities — min traffic as a non-increasing step
function of SMEM. The sibling area scripts use it; this program only needs a single
capacity point.)

---

## 3. Method 2 — latency-aware roofline

Classic roofline: `attainable_perf = min(compute_roof, OI · BW)`, `time = ops / perf`,
where `OI = ops / T`. Equivalent time form:

```
time = max( ops / compute_roof ,  T / BW )
     = max( compute_time      ,  memory_time )
```

The **latency-aware** extension replaces the physical bandwidth `BW` with an **effective**
bandwidth from **Little's law**, capturing that you must keep enough bytes in flight to
hide HBM latency:

```
latency  = HBM_LATENCY_CYCLES / clock_hz            # seconds
inflight = num_sm · C · W                            # bytes concurrently in flight (chip)
BW_eff   = min( BW_physical ,  inflight / latency )
memory_time = T / BW_eff
```

- `W = buffer_bytes` is the per-stage working set; `C = num_stages` is the software
  pipeline depth (how many stages' worth of tiles a threadblock keeps outstanding).
- **Why `num_sm`:** each of the GPU's `num_sm` SMs runs a threadblock that keeps `C` stages
  (each `W` bytes) outstanding, so chip-level in-flight bytes = `num_sm · C · W`. (The
  sibling area studies use `num_sm = 1`, modeling one big SM and dividing `BW` by
  `num_sm`; multiplying both the in-flight term and the physical `BW` by `num_sm` is the
  physically-correct chip-level form.)
- **`N = W` (design note carried from the area study):** the Little's-law "in-flight bytes
  per task" equals the *full* one-stage working set `W = buffer_bytes` (inputs **and**
  output tile), not just the input tiles. This was an explicit correction in the project.

### 3.1 Choosing the pipeline depth `C`

- **Feasibility:** a threadblock's `C` stages must fit its SMEM budget:
  `C · W ≤ SMEM_per_block`. Max feasible `C_max = floor(SMEM_per_block / W)`.
- **Saturation:** `BW_eff` reaches `BW_physical` once `num_sm·C·W/latency ≥ BW_physical`,
  i.e. at `C_sat = ceil( BW_physical · latency / (num_sm · W) )`.
- **Smallest-optimal depth** (used when the caller doesn't pin `C`):
  `C_best = min( C_max, max(C_sat, 1) )` — the fewest stages that saturate BW without
  exceeding SMEM. Pipelining beyond BW saturation buys nothing in this model.
- For large decode GEMMs the latency is trivially hidden (`W` is tens of KiB, `num_sm=24`),
  so `C_best` is usually **1**.

### 3.2 Final time

```
compute_time = ops / peak_tensor_flops
memory_time  = T / BW_eff
time         = max(compute_time, memory_time)
bottleneck   = "compute" | "memory" | "balanced"
```

---

## 4. GPU model & specifications (RTX 4060 Laptop)

The default `GpuModel` instance `RTX4060_LAPTOP` (AD107, Ada, compute capability **8.9**).
Provenance of each constant is important for anyone re-targeting the model.

### 4.1 Queried live from the device (authoritative)

Via `torch.cuda.get_device_properties(0)` on the actual machine:

| property | value | used as |
|---|---|---|
| `multi_processor_count` | **24** | `num_sm` |
| `shared_memory_per_multiprocessor` | **102400 B (100 KiB)** | `smem_per_sm_bytes` (context) |
| `shared_memory_per_block_optin` | **101376 B (99 KiB)** | `smem_per_block_bytes` (the `C·W` cap) |
| `shared_memory_per_block` (default) | 49152 B (48 KiB) | (not used; opt-in is the real cap) |
| `max_threads_per_multi_processor` | 1536 | context |
| `regs_per_multiprocessor` | 65536 | context |
| `L2_cache_size` | 33554432 (32 MiB) | context |
| `major.minor` | 8.9 (Ada) | ⇒ 4 tensor cores/SM |
| `total_memory` | 8585216000 (~8 GB) | context |
| max SM clock (`nvidia-smi --query-gpu=clocks.max.sm`) | **3105 MHz** | `clock_hz` |

### 4.2 Spec-sheet / derived constants (editable)

- **`tensor_cores = 96`** — Ada has 4 fourth-gen tensor cores per SM × 24 SM.
- **`tensor_flops_per_core_per_clock = 512`** — dense FP16 with FP32 accumulate. Derived
  from the A100 datasheet: 312 TFLOPS / (432 tensor cores × 1.41 GHz) = 512 FLOP/clock/core;
  the per-core rate is the same architecture family and matches the `512·10⁹` placeholder
  used throughout the sibling area scripts. **Sparse/other precisions differ** — change this
  if you model INT8, sparsity, or FP8.
- **`peak_tensor_flops = tensor_cores · 512 · clock_hz`** = 96·512·3.105e9 = **152.62
  TFLOP/s** (theoretical, at max boost).
- **`bw_bytes_per_s = 256e9`** — GDDR6, 128-bit bus, 16 Gbps effective ⇒ 256 GB/s.
- **`hbm_latency_cycles = 500`** — placeholder carried from the A100-based area study;
  `latency = 500 / 3.105e9 = 161 ns`. Because it is trivially hidden here, results are
  insensitive to it.
- **`bytes_per_element = 2`** (BF16/FP16).

### 4.3 Important caveat — laptop clock

The 3105 MHz is the hardware ceiling. A TGP-limited **laptop** AD107 will **not sustain**
it under a tensor load (sustained boost ~1.9–2.4 GHz depending on power limit). So
`peak_tensor_flops` is optimistic for **compute-bound** GEMMs. The decode-FFN GEMMs are
**memory-bound**, so this does not affect the target regime. For compute-bound cases pass a
realistic sustained clock via `--clock-mhz`. (In validation, a 4096³ GEMM measured
~19.5 TFLOP/s achieved — well under the 152 theoretical — confirming heavy laptop
throttling; irrelevant to memory-bound decode.)

---

## 5. Implementation walkthrough (`gemm_time_estimator.py`)

Pure Python, **no numpy** (so it runs as-is in the `profiling` conda env, which lacks
numpy). Only dependency is `snowcat_demo.model`. Structure:

### 5.1 `GpuModel` (frozen dataclass)

Fields: `name, num_sm, tensor_cores, tensor_flops_per_core_per_clock, clock_hz,
bw_bytes_per_s, smem_per_block_bytes, smem_per_sm_bytes, hbm_latency_cycles,
bytes_per_element=2`. Properties:
- `peak_tensor_flops = tensor_cores · tensor_flops_per_core_per_clock · clock_hz`
- `latency_seconds = hbm_latency_cycles / clock_hz`

`RTX4060_LAPTOP` is the instance from §4; `GPUS = {"rtx4060-laptop": RTX4060_LAPTOP}` maps
CLI names to instances (add new GPUs here).

### 5.2 `Mapping` (frozen dataclass)

`bm, bn, bk, loop_order: tuple[str,str,str], num_stages: int | None = None`. `num_stages =
None` ⇒ auto-pick `C_best`.

### 5.3 Helpers

- `optimal_mapping(m, n, k, gpu) -> Mapping` — `best_at_capacity(enumerate_mappings(...),
  gpu.smem_per_block_bytes)`; returns the min-traffic mapping that fits SMEM. This is the
  only place that *searches* the mapspace.
- `parse_loop_order(text) -> tuple` — accepts `"MKN"`, `"M-K-N"`, `"M,K,N"`; validates it's
  a permutation of `M,K,N` and one of `LOOP_ORDERS`.
- `_auto_num_stages(gpu, w) -> (C_best, C_max)` — implements §3.1. Returns `(0,0)` if even
  `C=1` doesn't fit (`W > SMEM_per_block`).

### 5.4 `estimate_gemm_time(m, n, k, mapping, gpu=RTX4060_LAPTOP) -> Estimate`

Steps:
1. **Validate tiles divide dims** (`M%BM==0`, etc.); else `ValueError` listing valid
   divisors.
2. Build `GemmWorkload(m=m, k=k, n=n, bytes_per_element=gpu.bytes_per_element)` and call
   `estimate_mapping_traffic(workload, bm, bk, bn, loop_order)` → `W = buffer_bytes`,
   `T = total_bytes`. Compute `ops`, `OI = ops/T`.
3. **Pick `C`:** if `mapping.num_stages is None`, use `_auto_num_stages` (`C_best`; if 0,
   fall back to 1 and add a "does not fit" note). Else use the pinned value (must be ≥1).
4. **Feasibility:** `fits_smem = C·W ≤ SMEM_per_block`; if not, append a note (the estimate
   is still computed, so you can explore infeasible mappings, but they're flagged).
5. **BW_eff:** `inflight = num_sm·C·W`; `BW_eff = min(bw_physical, inflight/latency)`.
6. **Roofline:** `compute_time = ops/peak_tensor_flops`; `memory_time = T/BW_eff`;
   `time = max(...)`; set `bottleneck`.
7. **Wave-quantization diagnostics** (see §6).
8. Return an `Estimate` (see §5.5).

### 5.5 `Estimate` (dataclass) — fields returned

Inputs echoed (`m,n,k,mapping,gpu`); traffic (`ops, working_set_bytes=W,
traffic_bytes=T, operational_intensity=OI`); pipeline (`num_stages=C,
max_feasible_stages=C_max, inflight_bytes, bw_eff_bytes_per_s`); roofline
(`compute_time_s, memory_time_s, time_s, bottleneck`); wave diagnostics (`output_tiles,
waves, sm_utilization, wave_adjusted_time_s`); `fits_smem: bool`; `notes: list[str]`.
Property `effective_tflops = ops/time_s/1e12`.

### 5.6 `format_estimate(e) -> str`

Human-readable multi-section report (traffic / pipeline / roofline / wave diagnostics /
notes). Times in ms, sizes in KiB/MiB.

### 5.7 CLI (`main` / `_parse_args`)

Flags: `--gpu` (key into `GPUS`), `--m/--n/--k`, `--bm/--bn/--bk`, `--order` (default
`MKN`), `--stages` (default auto), `--optimal` (ignore tile args, use `optimal_mapping`),
`--clock-mhz` (override `clock_hz` via `dataclasses.replace`), `--demo` (built-in
decode-FFN set `_DEMO_GEMMS`: router/up_gate/down for 4096-batched GLM-5.2). Requires
`m,n,k` (+ tiles unless `--optimal`).

Example commands:
```
conda run -n profiling python gemm_time_estimator.py \
    --m 128 --n 4096 --k 6144 --bm 64 --bn 128 --bk 64 --order MKN --stages 2
conda run -n profiling python gemm_time_estimator.py --m 128 --n 4096 --k 6144 --optimal
conda run -n profiling python gemm_time_estimator.py --demo
```

---

## 6. Wave-quantization diagnostic (not in the headline time)

Pure roofline assumes the GEMM saturates the whole chip. For decode FFN, `M` is tiny
(e.g. 128 per expert), so the output-tile count can be `< num_sm`, leaving SMs idle. The
estimator reports (but by default does **not** fold into `time_s`):

```
output_tiles   = (M/BM) · (N/BN)
waves          = ceil(output_tiles / num_sm)
sm_utilization = output_tiles / (waves · num_sm)
wave_adjusted_time = max( compute_time / sm_utilization , memory_time )
```

Only the compute roof is quantized by whole waves; memory time is left as-is. Since the
target GEMMs are memory-bound, `wave_adjusted_time` usually equals `time_s`. If you decide
wave quantization should affect the headline number (e.g. for compute-bound small GEMMs),
promote `wave_adjusted_time_s` to `time_s` — but note this is an approximation (it ignores
that memory traffic of a partial wave is also lower, and that tail SMs may overlap).

---

## 7. Validation (`validate_estimator.py`)

Optional script (needs GPU + torch) comparing the estimate — using the **snowcat-optimal
mapping** (`optimal_mapping`), so it's comparable to a cuBLAS kernel — against a real timed
FP16 `torch.matmul` (CUDA events, warmup + averaged iters). Measured on the actual RTX 4060
Laptop:

| gemm | M×N×K | tile (BM,BN,BK) | est ms | meas ms | est/meas | bottleneck |
|---|---|---|---:|---:|---:|---:|
| router | 4096×256×6144 | (128,256,1) | 0.598 | 0.75 | 0.80 | memory |
| up_gate | 128×4096×6144 | (128,256,1) | 0.299 | 0.41 | 0.72 | memory |
| down | 128×6144×2048 | (128,384,1) | 0.137 | 0.15 | 0.92 | memory |
| square2k | 2048×2048×2048 | (128,256,1) | 0.819 | 0.86 | 0.96 | memory |
| big | 4096×4096×4096 | (128,256,1) | 6.42 | 7.02 | 0.92 | memory |

**Interpretation:** the estimate is a consistent **optimistic lower bound** (72–96 % of
measured), exactly as expected for a roofline assuming peak BW and zero overhead. All
decode GEMMs are memory-bound — the target regime is well captured. The optimal mappings
favor `BK=1`, output-stationary (`k_inside_output=True`), which minimizes HBM traffic by
writing each output tile once; real kernels use larger `BK` but achieve near-minimum
traffic via L2, so the comparison is fair.

---

## 8. Assumptions & limitations (know these before trusting a number)

1. **Analytic single-entry-buffer traffic.** Real caches (L2 = 32 MiB here) can reduce
   traffic below the model, or fragmentation can raise it. The model is dataflow-idealized.
2. **Peak BW / peak clock.** No BW-efficiency or sustained-clock derating. Expect the true
   time to be higher (see §7). Multiply by an empirical efficiency (~1.1–1.4×) if you want
   an expected rather than best-case time.
3. **Compute roof is theoretical boost.** Wrong for a throttled laptop under compute-bound
   loads; use `--clock-mhz`. Irrelevant for memory-bound decode.
4. **Mapping is an input.** `estimate_gemm_time` does not check the mapping is good — only
   that tiles divide dims and (softly) that `C·W` fits SMEM. Garbage mapping → garbage
   (but self-consistent) time. Use `--optimal` for the best-case mapping.
5. **Divisor tiles only.** Snowcat enumerates divisor tile sizes; non-divisor tiles are
   rejected. Real kernels pad; not modeled.
6. **`num_stages` semantics** are the pipeline depth for latency hiding, not necessarily a
   specific library's `num_stages` (though they align conceptually).
7. **One GEMM, full-chip.** Concurrent kernels, grid-quantization tails beyond the §6
   diagnostic, and inter-kernel effects are out of scope.

---

## 9. How to extend

- **Add a GPU:** construct a new `GpuModel(...)` and register it in `GPUS`. Get `num_sm` and
  SMEM sizes from `torch.cuda.get_device_properties`; `clock_hz` from `nvidia-smi
  --query-gpu=clocks.max.sm`; bandwidth from the memory bus×data-rate; `tensor_cores` from
  (SM count × cores/SM for the arch); keep `tensor_flops_per_core_per_clock=512` for dense
  FP16/FP32-accumulate on Volta+…Ada, or recompute for other precisions.
- **Different precision:** change `bytes_per_element` (traffic) **and**
  `tensor_flops_per_core_per_clock` (compute roof) together; they are independent knobs.
- **Fold wave quantization into the headline:** see §6.
- **Fused epilogue (SwiGLU/RMS-scale) support:** add a CUDA-core roof
  `cuda_ops/cuda_roof` term to the `max(...)` (mirrors the sibling `ffn_fused_area*.py`
  model: `time = max(tensor_ops/tensor_roof, cuda_ops/cuda_roof, T/BW_eff)`), and add the
  epilogue's auxiliary state to `W`.
- **Search over mappings** (Option B): enumerate `enumerate_mappings` and evaluate
  `estimate_gemm_time` for each (optionally jointly with `C`) to find the true optimum,
  since a higher-traffic tiling with more favorable `W` can occasionally beat the Pareto
  point on `BW_eff`. `optimal_mapping` currently only takes the min-traffic Pareto point.
- **Latency sweep / multi-GPU tables:** the estimator is a pure function; wrap it in loops.

---

## 10. Environment & running

- Conda env **`profiling`** (Python 3.13, has `torch 2.7 cu128`, **no numpy** — keep the
  estimator numpy-free). Run via `conda run -n profiling python gemm_time_estimator.py …`.
- `torch` is used **only** by `validate_estimator.py` (measurement) and for one-time device
  introspection — **not** by the estimator itself.
- `snowcat_demo` must be importable (run from the project root, or set `PYTHONPATH` to it).
- Ignore the torch "Failed to initialize NumPy" warning — harmless; the estimator does not
  use numpy.

---

## 11. Units & conventions (quick reference)

- Bytes for all traffic/working-set quantities; FLOP for `ops`; FLOP/s for rooflines;
  seconds internally, printed as ms; bandwidth in bytes/s (printed GB/s = /1e9).
- `1 MiB = 2²⁰ B`, `1 KiB = 2¹⁰ B` (binary, for sizes); clocks/FLOP/s use decimal (1e9).
- `W = buffer_bytes` (one pipeline stage). `T = total_bytes = backing_store_bytes` (HBM
  traffic). `C = num_stages`. `OI = ops/T`.
- **Constructor arg order gotchas:** `GemmWorkload(m, k, n, bpe)` and
  `estimate_mapping_traffic(workload, m0, k0, n0, order)` both take **K in the 2nd/3rd
  slot**. The program passes `bm, bk, bn` (not `bm, bn, bk`) into the traffic call.
