"""Optimistic latency-aware roofline estimator — algorithmic-minimum traffic.

A sibling of gemm_time_estimator.py with the Snowcat / Orojenesis traffic model
REMOVED. Instead of computing the mapping-dependent HBM traffic (with tile re-reads
and the L2 reuse-distance refinement), this version uses the *algorithmic minimum*
traffic of a GEMM:

    T_min = (M*K + K*N + M*N) * bytes_per_element        # read A, read B, write C once
          + 2*(S-1)*M*N*accum_bytes   (if split_k S > 1) # split-K partial reduction

This is the smallest HBM traffic any GEMM implementation could possibly move
(perfect operand reuse, i.e. an infinite cache), so it MAXIMIZES operational
intensity and yields a deliberately OPTIMISTIC lower-bound time. It is
mapping-independent for a fixed (M,N,K), so loop order has no effect on traffic
(the `--order` flag is accepted but ignored), and every tile shares the same T.

Everything else is kept identical to the calibrated model: the latency-aware
saturating bandwidth (min(peak, per-SM-rate * active_SMs)), occupancy /
wave-quantization on both roofs, the compute roof, split-K's occupancy effect, and
the num_stages >= 2 floor. GPU profiles, Mapping, Estimate and the formatter are
reused from gemm_time_estimator so the two estimators stay in lockstep.

Usage mirrors the base tool:
  python gemm_time_estimator_min.py --m 128 --n 4096 --k 4096 --bm 64 --bn 64 --bk 32
  python gemm_time_estimator_min.py --m 128 --n 4096 --k 4096 --optimal   # time search
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace

from gemm_time_estimator import (
    GPUS,
    Estimate,
    Mapping,
    MIN_NUM_STAGES,
    OPTIMAL_MIN_BM,
    OPTIMAL_MIN_BN,
    OPTIMAL_MIN_BK,
    _auto_num_stages,
    format_estimate as _format_estimate_base,
)


def format_estimate(e: Estimate) -> str:
    """Base formatter with the snowcat section labels corrected for this model."""
    return (_format_estimate_base(e)
            .replace("-- snowcat traffic model --",
                     "-- ALGORITHMIC-MINIMUM traffic (optimistic; no snowcat) --")
            .replace("-- L2 model: DISABLED (raw snowcat traffic, L2 always miss) --",
                     "-- L2 model: not applicable (T already assumes perfect reuse) --"))


def algorithmic_min_traffic(m: int, n: int, k: int, bpe: int, accum_bytes: int,
                            split_k: int) -> tuple[int, int]:
    """(T, reduction_bytes): min HBM bytes = A+B read once, C written once, plus the
    split-K partial-reduction traffic (write+read of the S-1 extra partials)."""
    base = (m * k + k * n + m * n) * bpe
    reduction = 2 * (split_k - 1) * m * n * accum_bytes if split_k > 1 else 0
    return base + reduction, reduction


def estimate_gemm_time_min(m, n, k, mapping, gpu, occupancy_derate=True):
    """Optimistic estimate: algorithmic-minimum traffic + the kept latency-aware
    roofline. Returns an Estimate (l2 fields marked disabled/empty)."""
    for name, dim, tile in (("M", m, mapping.bm), ("N", n, mapping.bn), ("K", k, mapping.bk)):
        if tile <= 0 or dim % tile != 0:
            raise ValueError(f"tile {name}0={tile} must be a positive divisor of {name}={dim}")

    bpe = gpu.bytes_per_element
    ops = 2 * m * n * k
    w = (mapping.bm * mapping.bk + mapping.bk * mapping.bn + mapping.bm * mapping.bn) * bpe
    s = mapping.split_k
    if s < 1:
        raise ValueError("split_k must be >= 1")

    # ---- ALGORITHMIC-MINIMUM traffic (this is what replaces snowcat) -------- #
    t, reduction_bytes = algorithmic_min_traffic(m, n, k, bpe, gpu.accum_bytes, s)
    oi = ops / t
    notes = ["OPTIMISTIC model: T = algorithmic-minimum HBM traffic (perfect reuse); "
             "no snowcat re-reads, no L2 model. Real time is >= this."]
    k_tiles = k // mapping.bk
    if s > k_tiles:
        notes.append(f"split_k={s} exceeds K-tiles={k_tiles}; not realizable.")

    # ---- pipeline depth C (unchanged; MIN_NUM_STAGES floor) ----------------- #
    c_best_auto, c_max = _auto_num_stages(gpu, w)
    if mapping.num_stages is None:
        c = max(MIN_NUM_STAGES, c_best_auto)
        if c_best_auto == 0:
            notes.append(f"working set W={w} B exceeds SMEM/block={gpu.smem_per_block_bytes} B; "
                         f"even C=1 does not fit (using C={MIN_NUM_STAGES} floor anyway).")
    else:
        c = mapping.num_stages
        if c < MIN_NUM_STAGES:
            raise ValueError(f"num_stages must be >= {MIN_NUM_STAGES} (CUTLASS multistage floor)")
    fits_smem = c * w <= gpu.smem_per_block_bytes
    if not fits_smem:
        notes.append(f"C*W = {c*w} B exceeds SMEM/block = {gpu.smem_per_block_bytes} B "
                     f"(max feasible C = {c_max}).")

    # ---- occupancy / wave quantization (unchanged) ------------------------- #
    output_tiles = (m // mapping.bm) * (n // mapping.bn) * s
    active_sm = min(output_tiles, gpu.num_sm)
    waves = math.ceil(output_tiles / gpu.num_sm)
    sm_util = output_tiles / (waves * gpu.num_sm)

    # ---- latency-aware saturating bandwidth (unchanged) -------------------- #
    per_sm_bw = gpu.bw_bytes_per_s / gpu.bw_saturation_sms
    inflight = active_sm * c * w
    bw_latency = inflight / gpu.latency_seconds
    if occupancy_derate:
        bw_eff = min(gpu.bw_bytes_per_s, per_sm_bw * active_sm, bw_latency)
    else:
        bw_eff = gpu.bw_bytes_per_s

    # ---- roofline, both roofs occupancy-aware (unchanged) ------------------ #
    compute_time = ops / gpu.peak_tensor_flops
    compute_time_eff = compute_time / sm_util if (occupancy_derate and sm_util > 0) else compute_time
    memory_time = t / bw_eff
    time_s = max(compute_time_eff, memory_time)
    bottleneck = ("compute" if compute_time_eff > memory_time
                  else "memory" if memory_time > compute_time_eff else "balanced")

    return Estimate(
        m=m, n=n, k=k, mapping=mapping, gpu=gpu,
        ops=ops, working_set_bytes=w, traffic_bytes=int(round(t)),
        operational_intensity=oi, split_k=s, reduction_bytes=reduction_bytes,
        l2_enabled=False, l2_capacity_eff_bytes=0.0, l2_breakdown=[],
        num_stages=c, max_feasible_stages=c_max,
        inflight_bytes=inflight, bw_latency_bytes_per_s=bw_latency,
        occupancy_factor=bw_eff / gpu.bw_bytes_per_s, bw_eff_bytes_per_s=bw_eff,
        compute_time_s=compute_time, memory_time_s=memory_time,
        time_s=time_s, bottleneck=bottleneck,
        output_tiles=output_tiles, waves=waves, sm_utilization=sm_util,
        wave_adjusted_time_s=time_s, fits_smem=fits_smem,
        active_sm=active_sm, compute_time_eff_s=compute_time_eff, notes=notes,
    )


def _divisors_ge(x: int, lo: int):
    return [d for d in range(1, x + 1) if x % d == 0 and d >= min(lo, x)]


def optimal_mapping_min_by_time(m, n, k, gpu, *, min_bm=OPTIMAL_MIN_BM, min_bn=OPTIMAL_MIN_BN,
                                min_bk=OPTIMAL_MIN_BK, split_k=1):
    """Min estimated-time tiling over tensor-sensible tiles (loop order is irrelevant
    here since T is mapping-independent). Returns (Mapping, Estimate)."""
    best_e, best_map = None, None
    for bm in _divisors_ge(m, min_bm):
        for bn in _divisors_ge(n, min_bn):
            for bk in _divisors_ge(k, min_bk):
                cand = Mapping(bm=bm, bn=bn, bk=bk, loop_order=("M", "N", "K"),
                               num_stages=None, split_k=split_k)
                e = estimate_gemm_time_min(m, n, k, cand, gpu)
                if not e.fits_smem:
                    continue
                if best_e is None or e.time_s < best_e.time_s:
                    best_e = e
                    best_map = replace(cand, num_stages=e.num_stages)
    if best_map is None:
        raise ValueError(f"no tile with BM>={min_bm}, BN>={min_bn}, BK>={min_bk} fits SMEM")
    return best_map, best_e


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimistic (algorithmic-min-traffic) GEMM estimator")
    p.add_argument("--gpu", choices=sorted(GPUS), default="rtx4060-measured")
    p.add_argument("--m", type=int); p.add_argument("--n", type=int); p.add_argument("--k", type=int)
    p.add_argument("--bm", type=int); p.add_argument("--bn", type=int); p.add_argument("--bk", type=int)
    p.add_argument("--order", default="MNK", help="accepted but IGNORED (T is loop-order-independent here)")
    p.add_argument("--stages", type=int, default=None, help="pipeline depth C, >=2 (default: auto, floored at 2)")
    p.add_argument("--splitk", type=int, default=1, help="split-K slices S (adds partial-reduction traffic)")
    p.add_argument("--optimal", action="store_true",
                   help="search min estimated TIME over BM,BN>=64 (BK>=32) tiles")
    p.add_argument("--min-bm", type=int, default=OPTIMAL_MIN_BM)
    p.add_argument("--min-bn", type=int, default=OPTIMAL_MIN_BN)
    p.add_argument("--min-bk", type=int, default=OPTIMAL_MIN_BK)
    p.add_argument("--no-occupancy-bw", dest="occupancy_bw", action="store_false", default=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    gpu = GPUS[args.gpu]
    if args.stages is not None and args.stages < MIN_NUM_STAGES:
        raise SystemExit(f"--stages must be >= {MIN_NUM_STAGES} (CUTLASS multistage floor)")
    if args.m is None or args.n is None or args.k is None:
        raise SystemExit("need --m --n --k")

    if args.optimal:
        mapping, e = optimal_mapping_min_by_time(
            args.m, args.n, args.k, gpu,
            min_bm=args.min_bm, min_bn=args.min_bn, min_bk=args.min_bk, split_k=args.splitk)
        print(format_estimate(e))
        print(f"  [OPTIMISTIC estimator-optimal by TIME over BM>={args.min_bm}, "
              f"BN>={args.min_bn}, BK>={args.min_bk}:  {mapping.bm}x{mapping.bn}x{mapping.bk}  "
              f"stages={mapping.num_stages}]")
        return

    if None in (args.bm, args.bn, args.bk):
        raise SystemExit("need --bm --bn --bk (or --optimal)")
    mapping = Mapping(bm=args.bm, bn=args.bn, bk=args.bk, loop_order=("M", "N", "K"),
                      num_stages=args.stages, split_k=args.splitk)
    e = estimate_gemm_time_min(args.m, args.n, args.k, mapping, gpu, occupancy_derate=args.occupancy_bw)
    print(format_estimate(e))


if __name__ == "__main__":
    main()
