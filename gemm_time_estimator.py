"""Latency-aware Snowcat-roofline execution-time estimator for a single GEMM.

Estimates the wall-clock time of one GEMM on a real GPU (RTX 4060 Laptop by default)
given the GEMM size and a fully specified mapping (tile size, software-pipelining
stages, loop order).  No GEMM is run on the GPU -- the estimate is analytical.

Two tools, reused from the area studies (see notes/single_gemm_estimator.md):

  1. Snowcat / Orojenesis traffic model  (snowcat_demo.model.traffic)
       W = buffer_bytes  -> one-stage SMEM working set of the tiling
       T = total_bytes   -> minimum HBM backing-store traffic of the tiling
  2. Latency-aware roofline
       latency  = HBM_LATENCY_CYCLES / CLOCK_HZ
       inflight = num_sm * C * W                       (Little's law, chip level)
       BW_eff   = min(BW_physical, inflight / latency)
       time     = max(ops / peak_tensor_flops,  T / BW_eff)

Usage:
  conda run -n profiling python gemm_time_estimator.py \
      --m 128 --n 4096 --k 6144 --bm 64 --bn 128 --bk 64 --order MKN --stages 2

  # auto-pick the smallest-optimal pipeline depth C:
  conda run -n profiling python gemm_time_estimator.py --m 128 --n 4096 --k 6144 \
      --bm 64 --bn 128 --bk 64 --order MKN

  # model a split-K kernel (S slices) to recover occupancy on skinny problems:
  conda run -n profiling python gemm_time_estimator.py --m 128 --n 4096 --k 6144 \
      --bm 128 --bn 128 --bk 32 --order MNK --stages 3 --splitk 3

  # run the built-in decode-FFN example set:
  conda run -n profiling python gemm_time_estimator.py --demo
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, replace

from snowcat_demo.model.mapping import enumerate_mappings
from snowcat_demo.model.pareto import best_at_capacity
from snowcat_demo.model.traffic import LOOP_ORDERS, estimate_mapping_traffic
from snowcat_demo.model.workload import GemmWorkload, divisors


# --------------------------------------------------------------------------- #
# GPU model                                                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GpuModel:
    """Fixed hardware description used by the roofline.

    All fields are spec-sheet / device-queried constants -- edit them to model a
    different GPU.  Derived rooflines are computed as properties.
    """

    name: str
    num_sm: int
    tensor_cores: int
    tensor_flops_per_core_per_clock: float  # dense FP16/BF16 with FP32 accumulate
    clock_hz: float                         # SM clock used for both compute + latency
    bw_bytes_per_s: float                   # physical HBM/GDDR bandwidth (whole chip)
    smem_per_block_bytes: int               # usable shared memory for one threadblock
    smem_per_sm_bytes: int                  # total shared memory per SM (context)
    hbm_latency_cycles: float               # round-trip global-memory latency
    bytes_per_element: int = 2              # BF16 / FP16
    accum_bytes: int = 4                    # split-K partial-sum precision (FP32)
    l2_bytes: int = 32 * 1024 * 1024        # physical L2 capacity (device-queried)
    l2_capacity_alpha: float = 0.6          # effective fraction of L2 usable for reuse
                                            # (absorbs associativity + concurrency +
                                            # slice non-uniformity; calibrate via ncu)
    bw_saturation_sms: float = 5.0          # # of active SMs at which the DRAM bus
                                            # saturates (bw_eff = min(peak, active/this
                                            # * peak)); measured via occupancy_bw.cu.

    @property
    def peak_tensor_flops(self) -> float:
        """Chip-wide tensor-core compute roof (FLOP/s)."""
        return self.tensor_cores * self.tensor_flops_per_core_per_clock * self.clock_hz

    @property
    def latency_seconds(self) -> float:
        return self.hbm_latency_cycles / self.clock_hz


# RTX 4060 Laptop GPU (AD107, Ada, compute capability 8.9).
# SM count / SMEM sizes queried live from torch.cuda.get_device_properties(0);
# clock from `nvidia-smi --query-gpu=clocks.max.sm`; tensor rate + bandwidth from
# the Ada spec sheet.  See notes/single_gemm_estimator.md for provenance.
RTX4060_LAPTOP = GpuModel(
    name="NVIDIA GeForce RTX 4060 Laptop GPU",
    num_sm=24,
    tensor_cores=96,                          # 4 4th-gen tensor cores per SM
    tensor_flops_per_core_per_clock=512.0,    # dense FP16, FP32 accumulate (A100-derived)
    clock_hz=3105e6,                          # max SM boost; laptop sustains less (editable)
    bw_bytes_per_s=256e9,                      # GDDR6, 128-bit, 16 Gbps effective
    smem_per_block_bytes=101376,              # torch shared_memory_per_block_optin
    smem_per_sm_bytes=102400,                 # torch shared_memory_per_multiprocessor
    # Measured L2-miss (VRAM->SM) dependent-load latency via pointer-chase
    # microbenchmark (512 MiB working set >> 32 MiB L2): ~282 ns, i.e. ~600 SM
    # cycles at the ~2.1 GHz the SM actually sustained during the test. Expressed
    # at this model's 3.105 GHz reference clock the same 282 ns is ~876 cycles
    # (latency_seconds = 876 / 3.105e9 = 282 ns). Replaces the 500-cycle/161 ns
    # A100 placeholder. See vramLatency.cu p-chase sweep.
    hbm_latency_cycles=876,                   # measured DRAM latency (L2 always miss)
    bytes_per_element=2,
    l2_bytes=33554432,                        # 32 MiB, torch get_device_properties L2
)

# Calibrated to THIS device by direct microbenchmark (membench.cu, gpu_boost_probe.cu,
# occupancy_bw.cu), with the GPU WARMED under continuous load so it boosts out of its
# idle underclock (nvidia-smi confirms 210 MHz/6 W idle -> ~1680 MHz/35 W under load):
#   * SM clock       : ~1680 MHz under a sustained tensor GEMM (pinned at the 35 W cap;
#                      ~2070 MHz for lighter FP32 loads). NOT 3105 (max-boost) and NOT
#                      the earlier 372 MHz artifact (that was measured cold/idle).
#   * tensor rate    : ~128 FLOP/clk/core for FP16->FP32 on consumer Ada (GeForce halves
#                      the FP32-accumulate rate vs the 512 A100 figure). 96 cores * 128 *
#                      1.70 GHz = 20.9 TFLOP/s roof; measured cuBLAS 4096^3 warm = 23.4
#                      TFLOP/s (~92% of the ~2.0 GHz peak).
#   * DRAM bandwidth : 218 GB/s achievable (read-stream, 512 MiB >> L2) vs 256 spec.
#   * DRAM latency   : 301 ns (pointer-chase) = 512 cycles @ 1.70 GHz.
RTX4060_MEASURED = replace(
    RTX4060_LAPTOP,
    name="RTX 4060 Laptop (measured/calibrated)",
    clock_hz=1500e6,
    tensor_flops_per_core_per_clock=128.0,
    bw_bytes_per_s=218e9,
    hbm_latency_cycles=512,
)

GPUS: dict[str, GpuModel] = {
    "rtx4060-laptop": RTX4060_LAPTOP,
    "rtx4060-measured": RTX4060_MEASURED,
}

# L2/L1 lines are 128 B = 4 x 32 B sectors; DRAM traffic is counted in 32 B sectors
# actually touched, so accesses that under-fill a sector inflate real DRAM bytes.
SECTOR_BYTES = 32


# --------------------------------------------------------------------------- #
# Mapping                                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Mapping:
    """A fully specified GEMM tiling.

    bm/bn/bk are the tile sizes (m0/n0/k0 in Snowcat terms) and must divide M/N/K.
    loop_order is the tile-loop nesting, outermost first, e.g. ("M", "K", "N").
    num_stages is the software-pipeline depth C; None -> auto-pick smallest optimal.
    split_k is the number of K-slices S (1 = disabled); partitions the K reduction
    across S threadblocks per output tile, then reduces the S partial sums.
    """

    bm: int
    bn: int
    bk: int
    loop_order: tuple[str, str, str]
    num_stages: int | None = None
    split_k: int = 1


# Tensor-core MMA instruction-shape lower bounds (Ada 4th-gen, 16x8x16 for
# FP16/BF16).  A threadblock tile smaller than one MMA cannot be issued to the
# tensor cores, so the --optimal search is restricted to tiles a real tensor-core
# kernel could actually run.  Without this the unconstrained snowcat optimum
# drives BK->1 (minimal buffer => minimal traffic), which is not realizable.
MMA_MIN_BM = 16
MMA_MIN_BN = 8
MMA_MIN_BK = 16


def optimal_mapping(m: int, n: int, k: int, gpu: GpuModel) -> Mapping:
    """The snowcat min-HBM-traffic tiling that fits one threadblock's SMEM budget.

    Convenience for comparing against a library kernel (cuBLAS also picks a good
    mapping).  This is a *search* over the snowcat mapspace, unlike the rest of the
    estimator which takes the mapping as a given input.

    The search is restricted to MMA-legal tiles (BM>=16, BN>=8, BK>=16); the
    bounds are clamped to the problem size so tiny dims (e.g. M<16) stay feasible.
    """
    workload = GemmWorkload(m=m, k=k, n=n, bytes_per_element=gpu.bytes_per_element)
    min_bm, min_bn, min_bk = min(MMA_MIN_BM, m), min(MMA_MIN_BN, n), min(MMA_MIN_BK, k)
    points = [
        p for p in enumerate_mappings(workload)
        if p.mapping.m0 >= min_bm and p.mapping.n0 >= min_bn and p.mapping.k0 >= min_bk
    ]
    if not points:
        raise ValueError("no MMA-legal mapping (BM>=16, BN>=8, BK>=16) exists")
    best = best_at_capacity(points, gpu.smem_per_block_bytes)
    if best is None:
        raise ValueError("no MMA-legal mapping fits the SMEM budget")
    mp = best.mapping
    return Mapping(bm=mp.m0, bn=mp.n0, bk=mp.k0, loop_order=mp.loop_order)


def parse_loop_order(text: str) -> tuple[str, str, str]:
    """Accept 'MKN' or 'M-K-N' / 'M,K,N' -> ('M', 'K', 'N')."""
    cleaned = text.upper().replace("-", "").replace(",", "").replace(" ", "")
    order = tuple(cleaned)
    if len(order) != 3 or set(order) != {"M", "K", "N"}:
        raise ValueError(f"loop order must be a permutation of M,K,N; got {text!r}")
    if order not in LOOP_ORDERS:
        raise ValueError(f"unsupported loop order {order}; must be one of {LOOP_ORDERS}")
    return order  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# L2 reuse-distance model (Tier 1)                                              #
# --------------------------------------------------------------------------- #
# Each GEMM operand depends on exactly two of {M,N,K}; the omitted dim is the one
# it is re-fetched along ("reuse dim").  Row-major layout => the contiguous (last)
# axis is K for A[M,K] and N for W[K,N] / OUT[M,N].
_OPERAND_DIMS = {"A": ("M", "K"), "W": ("K", "N"), "OUT": ("M", "N")}
_REUSE_DIM = {"A": "N", "W": "M", "OUT": "K"}
_CONTIG_DIM = {"A": "K", "W": "N", "OUT": "N"}


def _reuse_distance_bytes(
    op: str, m: int, n: int, k: int, bm: int, bn: int, bk: int,
    loop_order: tuple[str, str, str], bpe: int,
) -> int:
    """Distinct bytes touched between two accesses to the same tile of `op`.

    The tile is re-fetched when its reuse dim advances; in between, the loops
    *inner* to that reuse dim sweep fully.  Each operand Y contributes the product
    of its two dims, using the full extent for dims that are inner loops and the
    tile extent for dims that are the reuse dim or outer (held fixed).
    """
    full = {"M": m, "N": n, "K": k}
    tile = {"M": bm, "N": bn, "K": bk}
    pos = {d: i for i, d in enumerate(loop_order)}
    inner = {d for d in ("M", "N", "K") if pos[d] > pos[_REUSE_DIM[op]]}
    total = 0
    for dims in _OPERAND_DIMS.values():
        f = bpe
        for d in dims:
            f *= full[d] if d in inner else tile[d]
        total += f
    return total


def _sector_inflation(op: str, bm: int, bn: int, bk: int, bpe: int) -> float:
    """DRAM-byte inflation from under-filled 32 B sectors (row-major operand).

    A contiguous run shorter than one sector still fetches a whole 32 B sector.
    """
    contig_bytes = {"A": bk, "W": bn, "OUT": bn}[op] * bpe
    return max(1.0, SECTOR_BYTES / contig_bytes)


def _l2_concurrency(op: str, mt: int, nt: int, num_sm: int) -> int:
    """Distinct co-resident tiles of `op` sharing L2 in one wave.

    The chip-shared L2 must hold every concurrently-running tile's working set at
    once, so a *private* operand (a distinct panel per concurrent tile) competes
    for capacity while a *shared/broadcast* operand (same data read by all the
    wave's tiles) needs holding only once.  Assuming a row-major tile raster
    (N fastest), one wave of P=min(mt*nt, num_sm) tiles spans `n_conc` distinct
    N-columns and `m_conc` distinct M-rows; A reuses along N so its distinct copies
    scale with M-rows, W along M so with N-columns.
    """
    p = min(mt * nt, num_sm)
    n_conc = min(p, nt)
    m_conc = max(1, math.ceil(p / nt))
    return {"A": m_conc, "W": n_conc, "OUT": p}[op]


# --------------------------------------------------------------------------- #
# Estimation                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class Estimate:
    # inputs
    m: int
    n: int
    k: int
    mapping: Mapping
    gpu: GpuModel
    # snowcat traffic
    ops: int
    working_set_bytes: int          # W = buffer_bytes (one pipeline stage)
    traffic_bytes: int              # T = HBM traffic (incl. split-K reduction)
    operational_intensity: float    # OI = ops / T  (FLOP/byte)
    split_k: int                    # S = split-K slices (1 = disabled)
    reduction_bytes: int            # extra HBM traffic from split-K partial reduction
    l2_enabled: bool                # whether the L2 reuse-distance model was applied
    l2_capacity_eff_bytes: float    # C_eff = alpha * L2 (0 if disabled)
    # pipeline / latency
    num_stages: int                 # C actually used
    max_feasible_stages: int        # floor(SMEM_per_block / W)
    inflight_bytes: float           # num_sm * C * W
    bw_latency_bytes_per_s: float   # latency-capped BW before occupancy derate
    occupancy_factor: float         # SM-utilization factor applied to BW (1.0 = off)
    bw_eff_bytes_per_s: float       # achieved BW = bw_latency * occupancy_factor
    # roofline
    compute_time_s: float
    memory_time_s: float
    time_s: float
    bottleneck: str
    # wave-quantization diagnostics
    output_tiles: int
    waves: int
    sm_utilization: float
    wave_adjusted_time_s: float
    # feasibility
    fits_smem: bool
    active_sm: int = 0                 # concurrent CTAs = min(output_tiles, num_sm)
    compute_time_eff_s: float = 0.0    # compute_time / sm_util (idle-SM derate)
    # L2 per-operand breakdown: (op, cold_B, mult, reuse_dist_B, frac_cached, sector, dram_B)
    l2_breakdown: list = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def effective_tflops(self) -> float:
        return self.ops / self.time_s / 1e12 if self.time_s > 0 else float("nan")


def _auto_num_stages(gpu: GpuModel, w: int) -> tuple[int, int]:
    """Smallest-optimal pipeline depth C and the max feasible depth (notes model)."""
    c_max = gpu.smem_per_block_bytes // w
    if c_max < 1:
        return 0, 0
    # BW saturates when num_sm * C * W / latency >= bw
    c_sat = math.ceil(gpu.bw_bytes_per_s * gpu.latency_seconds / (gpu.num_sm * w))
    c_best = min(c_max, max(c_sat, 1))
    return c_best, c_max


def estimate_gemm_time(
    m: int,
    n: int,
    k: int,
    mapping: Mapping,
    gpu: GpuModel = RTX4060_LAPTOP,
    occupancy_derate: bool = True,
    l2: bool = True,
    l2_alpha: float | None = None,
    pin: tuple[str, ...] = (),
) -> Estimate:
    """Latency-aware Snowcat-roofline time estimate for one GEMM + mapping."""
    for name, dim, tile in (("M", m, mapping.bm), ("N", n, mapping.bn), ("K", k, mapping.bk)):
        if tile <= 0 or dim % tile != 0:
            raise ValueError(
                f"tile {name}0={tile} must be a positive divisor of {name}={dim}; "
                f"nearest divisors: {divisors(dim)}"
            )

    workload = GemmWorkload(m=m, k=k, n=n, bytes_per_element=gpu.bytes_per_element)
    traffic = estimate_mapping_traffic(
        workload, mapping.bm, mapping.bk, mapping.bn, mapping.loop_order
    )
    w = traffic.buffer_bytes          # one-stage working set W
    ops = workload.operations

    notes: list[str] = []

    # ---- split-K ----------------------------------------------------------- #
    # S slices each reduce K/S and emit an M*N partial in accumulator precision;
    # combining them costs an extra write+read of the (S-1) partial tiles beyond
    # the single final C write already counted in the base snowcat traffic.
    s = mapping.split_k
    if s < 1:
        raise ValueError("split_k must be >= 1")
    k_tiles = k // mapping.bk
    if s > k_tiles:
        notes.append(f"split_k={s} exceeds K-tiles={k_tiles}; not realizable.")
    elif k_tiles % s != 0:
        notes.append(
            f"split_k={s} does not evenly divide K-tiles={k_tiles}; slices are "
            "unbalanced (estimate assumes an even split)."
        )
    reduction_bytes = 2 * (s - 1) * m * n * gpu.accum_bytes

    # ---- L2 reuse-distance model ------------------------------------------ #
    # snowcat gives each operand's total read traffic = cold_footprint x re-read
    # multiplier.  L2 absorbs re-reads whose reuse distance fits in effective L2,
    # collapsing the multiplier toward 1.  `--no-l2` (l2=False) keeps the raw
    # snowcat total (exact legacy "L2 always miss" behaviour).
    bpe = gpu.bytes_per_element
    cold = {"A": m * k * bpe, "W": k * n * bpe, "OUT": m * n * bpe}
    snow = {
        "A": traffic.a_read_bytes,
        "W": traffic.w_read_bytes,
        "OUT": traffic.b_read_bytes + traffic.b_write_bytes,
    }
    alpha = gpu.l2_capacity_alpha if l2_alpha is None else l2_alpha
    c_eff = alpha * gpu.l2_bytes
    mt, nt = m // mapping.bm, n // mapping.bn      # output-tile grid (excl. split-K)
    l2_breakdown: list[tuple] = []
    if l2:
        operand_dram = 0.0
        for op in ("A", "W", "OUT"):
            c = cold[op]
            mult = snow[op] / c if c > 0 else 1.0
            conc = _l2_concurrency(op, mt, nt, gpu.num_sm)
            if mult > 1.0:
                rd = float(_reuse_distance_bytes(
                    op, m, n, k, mapping.bm, mapping.bn, mapping.bk,
                    mapping.loop_order, bpe))
                # chip-shared L2: `conc` distinct copies of this reuse working set
                # must coexist, so effective capacity per stream is C_eff / conc.
                frac = 1.0 if op in pin else min(1.0, c_eff / (rd * conc))
            else:
                rd, frac = 0.0, 1.0
            miss = 1.0 + (mult - 1.0) * (1.0 - frac)
            sect = _sector_inflation(op, mapping.bm, mapping.bn, mapping.bk, bpe)
            dram = c * miss * sect
            operand_dram += dram
            l2_breakdown.append((op, c, mult, rd, conc, frac, sect, dram))
        if pin:
            notes.append(f"L2-pinned operands (persistence): {', '.join(pin)}")
    else:
        operand_dram = float(traffic.total_bytes)   # legacy: no L2, no sector model

    t = operand_dram + reduction_bytes
    oi = ops / t

    # ---- pipeline depth C -------------------------------------------------- #
    c_best_auto, c_max = _auto_num_stages(gpu, w)
    if mapping.num_stages is None:
        c = c_best_auto
        if c == 0:
            notes.append(
                f"working set W={w} B exceeds SMEM/block={gpu.smem_per_block_bytes} B; "
                "even C=1 does not fit."
            )
            c = 1
    else:
        c = mapping.num_stages
        if c < 1:
            raise ValueError("num_stages must be >= 1")

    fits_smem = c * w <= gpu.smem_per_block_bytes
    if not fits_smem:
        notes.append(
            f"C*W = {c * w} B exceeds SMEM/block = {gpu.smem_per_block_bytes} B "
            f"(max feasible C = {c_max}); pipeline would not fit on hardware."
        )

    # ---- occupancy / wave quantization ------------------------------------ #
    # Split-K launches S concurrent slices per output tile, multiplying the grid.
    output_tiles = (m // mapping.bm) * (n // mapping.bn) * s
    active_sm = min(output_tiles, gpu.num_sm)      # CTAs resident at once (cap = chip)
    waves = math.ceil(output_tiles / gpu.num_sm)
    sm_util = output_tiles / (waves * gpu.num_sm)  # avg fraction of SMs busy

    # ---- effective bandwidth: MEASURED saturating law, NOT bw*sm_util ------ #
    # occupancy_bw.cu (this device) shows bw_eff is a clamped ramp, not linear:
    # one active SM injects ~bw_peak/bw_saturation_sms, and the DRAM bus saturates
    # once ~bw_saturation_sms SMs are active (~5 here), staying flat to 100% occ.
    # So bw_eff = min(bw_peak, per_sm * active_sm): concave in occupancy. The old
    # bw*sm_util under-predicted BW across the whole mid-range (at 33% occ it said
    # 0.33*peak; the hardware already delivers ~full peak). Little's-law latency
    # cap kept as a secondary bound (binds only when per-SM MLP is tiny).
    per_sm_bw = gpu.bw_bytes_per_s / gpu.bw_saturation_sms
    inflight = active_sm * c * w
    bw_latency = inflight / gpu.latency_seconds
    if occupancy_derate:
        bw_eff = min(gpu.bw_bytes_per_s, per_sm_bw * active_sm, bw_latency)
    else:
        bw_eff = gpu.bw_bytes_per_s

    # ---- roofline: BOTH roofs are occupancy-aware ------------------------- #
    compute_time = ops / gpu.peak_tensor_flops
    # Idle SMs do no FLOPs, so tensor throughput scales LINEARLY with the busy-SM
    # fraction (SMs are independent, no shared bottleneck) -- unlike the saturating
    # DRAM bus. A 1-wave, 16-of-24-SM GEMM computes at 16/24 of peak.
    compute_time_eff = compute_time / sm_util if (occupancy_derate and sm_util > 0) else compute_time
    memory_time = t / bw_eff
    time_s = max(compute_time_eff, memory_time)
    if compute_time_eff > memory_time:
        bottleneck = "compute"
    elif memory_time > compute_time_eff:
        bottleneck = "memory"
    else:
        bottleneck = "balanced"
    wave_adjusted_time = time_s   # headline is already occupancy-aware on both roofs

    return Estimate(
        m=m, n=n, k=k, mapping=mapping, gpu=gpu,
        ops=ops, working_set_bytes=w, traffic_bytes=int(round(t)),
        operational_intensity=oi,
        split_k=s, reduction_bytes=reduction_bytes,
        l2_enabled=l2, l2_capacity_eff_bytes=(c_eff if l2 else 0.0),
        l2_breakdown=l2_breakdown,
        num_stages=c, max_feasible_stages=c_max,
        inflight_bytes=inflight, bw_latency_bytes_per_s=bw_latency,
        occupancy_factor=bw_eff / gpu.bw_bytes_per_s, bw_eff_bytes_per_s=bw_eff,
        compute_time_s=compute_time, memory_time_s=memory_time,
        time_s=time_s, bottleneck=bottleneck,
        output_tiles=output_tiles, waves=waves, sm_utilization=sm_util,
        wave_adjusted_time_s=wave_adjusted_time,
        active_sm=active_sm, compute_time_eff_s=compute_time_eff,
        fits_smem=fits_smem, notes=notes,
    )


# --------------------------------------------------------------------------- #
# Reporting                                                                     #
# --------------------------------------------------------------------------- #
def _format_l2_block(e: Estimate, mib: int) -> list[str]:
    """Per-operand L2 reuse-distance breakdown (or a one-line 'disabled' note)."""
    if not e.l2_enabled:
        return ["  -- L2 model: DISABLED (raw snowcat traffic, L2 always miss) --", ""]
    lines = [
        f"  -- L2 reuse-distance model (C_eff = {e.l2_capacity_eff_bytes / mib:.1f} MiB "
        f"= alpha x L2) --",
        f"  {'op':<4} {'cold':>9} {'reread':>7} {'reuse-dist':>11} {'conc':>5} "
        f"{'cached':>7} {'sector':>7} {'->DRAM':>10}",
    ]
    for op, cold, mult, rd, conc, frac, sect, dram in e.l2_breakdown:
        rd_str = f"{rd / mib:.2f}MiB" if rd > 0 else "-"
        lines.append(
            f"  {op:<4} {cold / mib:>7.2f}Mi {mult:>6.1f}x {rd_str:>11} {conc:>4}x "
            f"{frac * 100:>6.0f}% {sect:>6.2f}x {dram / mib:>8.2f}Mi"
        )
    return lines + [""]


def format_estimate(e: Estimate) -> str:
    mib = 2 ** 20
    lines = [
        f"=== GEMM {e.m}x{e.n}x{e.k}  on  {e.gpu.name} ===",
        f"  ops                 : {e.ops / 1e9:.3f} GFLOP  ({e.ops / 1e12:.4f} TFLOP)",
        f"  tile (BM,BN,BK)     : ({e.mapping.bm}, {e.mapping.bn}, {e.mapping.bk})  "
        f"loop_order={'-'.join(e.mapping.loop_order)}",
        "",
        "  -- snowcat traffic model --",
        f"  working set W       : {e.working_set_bytes / 1024:.2f} KiB  "
        f"({e.working_set_bytes} B)  per pipeline stage",
        f"  HBM traffic T       : {e.traffic_bytes / mib:.3f} MiB  ({e.traffic_bytes} B)",
        f"  split-K slices S    : {e.split_k}"
        + (f"   (+{e.reduction_bytes / mib:.3f} MiB partial-reduction traffic)"
           if e.split_k > 1 else "   (disabled)"),
        f"  op. intensity OI    : {e.operational_intensity:.3f} FLOP/byte",
        "",
        *_format_l2_block(e, mib),
        "  -- latency-aware pipeline --",
        f"  num_stages C        : {e.num_stages}"
        + (f"  (auto; max feasible = {e.max_feasible_stages})"
           if e.mapping.num_stages is None
           else f"  (max feasible = {e.max_feasible_stages})"),
        f"  SMEM for C stages   : {e.num_stages * e.working_set_bytes / 1024:.2f} KiB "
        f"/ {e.gpu.smem_per_block_bytes / 1024:.2f} KiB per block"
        + ("" if e.fits_smem else "   *** DOES NOT FIT ***"),
        f"  active SMs          : {e.active_sm} / {e.gpu.num_sm}  "
        f"(concurrent CTAs; bus saturates by ~{e.gpu.bw_saturation_sms:g} SMs)",
        f"  HBM latency         : {e.gpu.hbm_latency_cycles:g} cycles "
        f"({e.gpu.latency_seconds * 1e9:.1f} ns)",
        f"  effective BW        : {e.bw_eff_bytes_per_s / 1e9:.1f} GB/s "
        f"({e.occupancy_factor * 100:.0f}% of peak; saturating law min(peak "
        f"{e.gpu.bw_bytes_per_s / 1e9:.0f}, {e.gpu.bw_bytes_per_s / e.gpu.bw_saturation_sms / 1e9:.0f}/SM x "
        f"{e.active_sm}))",
        "",
        "  -- roofline (both roofs occupancy-aware) --",
        f"  compute roof        : {e.gpu.peak_tensor_flops / 1e12:.2f} TFLOP/s",
        f"  compute time        : {e.compute_time_s * 1e3:.4f} ms  "
        f"-> {e.compute_time_eff_s * 1e3:.4f} ms  (/ sm_util {e.sm_utilization:.3f}; idle-SM derate)",
        f"  memory time         : {e.memory_time_s * 1e3:.4f} ms",
        f"  >> time             : {e.time_s * 1e3:.4f} ms   "
        f"[{e.bottleneck}-bound]   {e.effective_tflops:.1f} TFLOP/s",
        "",
        "  -- wave quantization (diagnostic) --",
        f"  output tiles        : {e.output_tiles}  "
        f"(M/BM={e.m // e.mapping.bm} x N/BN={e.n // e.mapping.bn}"
        + (f" x S={e.split_k}" if e.split_k > 1 else "") + ")",
        f"  waves               : {e.waves}  (num_sm={e.gpu.num_sm})   "
        f"SM utilization = {e.sm_utilization * 100:.1f}%",
        f"  wave-adjusted time  : {e.wave_adjusted_time_s * 1e3:.4f} ms",
    ]
    for note in e.notes:
        lines.append(f"  NOTE: {note}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Built-in decode-FFN example set (GLM-5.2, 4096-batched, 128 tok/expert)       #
# --------------------------------------------------------------------------- #
_DEMO_GEMMS = [
    # (label, M, N, K, BM, BN, BK, loop_order, stages)
    # Illustrative (not traffic-optimized) mappings that fit the 99 KiB SMEM budget.
    # For the small-M decode GEMMs, BM = full M keeps the weight streamed once.
    ("router",  4096, 256, 6144, 128, 128, 64, "MKN", None),
    ("up_gate",  128, 4096, 6144, 128, 128, 64, "MKN", None),
    ("down",     128, 6144, 2048, 128, 128, 64, "MKN", None),
]


def run_demo(gpu: GpuModel, occupancy_derate: bool = True,
             l2: bool = True, l2_alpha: float | None = None) -> None:
    for label, m, n, k, bm, bn, bk, order, stages in _DEMO_GEMMS:
        mapping = Mapping(bm=bm, bn=bn, bk=bk,
                          loop_order=parse_loop_order(order), num_stages=stages)
        e = estimate_gemm_time(m, n, k, mapping, gpu, occupancy_derate=occupancy_derate,
                               l2=l2, l2_alpha=l2_alpha)
        print(f"\n### {label}")
        print(format_estimate(e))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", choices=sorted(GPUS), default="rtx4060-laptop")
    p.add_argument("--m", type=int, help="GEMM M")
    p.add_argument("--n", type=int, help="GEMM N")
    p.add_argument("--k", type=int, help="GEMM K")
    p.add_argument("--bm", type=int, help="tile BM (m0), must divide M")
    p.add_argument("--bn", type=int, help="tile BN (n0), must divide N")
    p.add_argument("--bk", type=int, help="tile BK (k0), must divide K")
    p.add_argument("--order", default="MKN", help="loop order, e.g. MKN or M-K-N")
    p.add_argument("--stages", type=int, default=None,
                   help="software-pipeline depth C (default: auto smallest-optimal)")
    p.add_argument("--splitk", type=int, default=1,
                   help="split-K slices S (default 1 = disabled); partitions the K "
                        "reduction across S threadblocks + a partial-sum reduction")
    p.add_argument("--optimal", action="store_true",
                   help="ignore --bm/--bn/--bk/--order and use the snowcat min-traffic "
                        "mapping that fits SMEM")
    p.add_argument("--clock-mhz", type=float, default=None,
                   help="override SM clock in MHz (e.g. a sustained laptop boost)")
    p.add_argument("--no-occupancy-bw", dest="occupancy_bw", action="store_false",
                   default=True,
                   help="disable the occupancy (SM-utilization) derate on effective "
                        "bandwidth; recovers the pure peak-BW roofline")
    p.add_argument("--no-l2", dest="l2", action="store_false", default=True,
                   help="disable the L2 reuse-distance model; use raw snowcat traffic "
                        "(exact legacy 'L2 always miss' behaviour)")
    p.add_argument("--l2-alpha", type=float, default=None,
                   help="override effective-L2 fraction alpha (C_eff = alpha x L2); "
                        "default from the GPU model, calibrate via l2_calibrate.py")
    p.add_argument("--pin", default="",
                   help="comma-separated operands to L2-pin (persistence): A,W,OUT")
    p.add_argument("--demo", action="store_true",
                   help="run the built-in decode-FFN example set")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    gpu = GPUS[args.gpu]
    if args.clock_mhz is not None:
        gpu = replace(gpu, clock_hz=args.clock_mhz * 1e6)

    if args.demo:
        run_demo(gpu, occupancy_derate=args.occupancy_bw,
                 l2=args.l2, l2_alpha=args.l2_alpha)
        return

    required = ("m", "n", "k") if args.optimal else ("m", "n", "k", "bm", "bn", "bk")
    missing = [f for f in required if getattr(args, f) is None]
    if missing:
        raise SystemExit(
            f"missing required args: {', '.join('--' + x for x in missing)} "
            "(or pass --demo)"
        )

    if args.optimal:
        mapping = optimal_mapping(args.m, args.n, args.k, gpu)
        mapping = replace(
            mapping,
            num_stages=args.stages if args.stages is not None else mapping.num_stages,
            split_k=args.splitk,
        )
    else:
        mapping = Mapping(
            bm=args.bm, bn=args.bn, bk=args.bk,
            loop_order=parse_loop_order(args.order), num_stages=args.stages,
            split_k=args.splitk,
        )
    pin = tuple(p.strip().upper() for p in args.pin.split(",") if p.strip())
    bad_pins = [p for p in pin if p not in _OPERAND_DIMS]
    if bad_pins:
        raise SystemExit(f"--pin operands must be among A,W,OUT; got {bad_pins}")
    e = estimate_gemm_time(args.m, args.n, args.k, mapping, gpu,
                           occupancy_derate=args.occupancy_bw,
                           l2=args.l2, l2_alpha=args.l2_alpha, pin=pin)
    print(format_estimate(e))


if __name__ == "__main__":
    main()
