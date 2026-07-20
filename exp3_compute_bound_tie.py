"""EXPERIMENT 3 — the compute-bound tie.

For w in {512,1024,2048}, L=6, M=131072, H100: is fuse-all EVER strictly worse than a
split in the compute-bound regime, and why? Print exact times (many decimals) of
fuse-all vs best split vs unfused, the true ratios, and the roofline occupancy
(out_tiles, sm_util, waves, active_sm, compute_eff, memory, bott) of fuse-all's single
big kernel vs a 2-segment split's two smaller kernels.

Everything is timed by the exact same harness (segment_time / analyze). We ONLY add an
occupancy read-out by replaying segment_time's inner roofline for its chosen m0.
"""
from __future__ import annotations

import math

from gemm_time_estimator import GPUS, optimal_mapping_by_time
from snowcat_demo.model.workload import divisors
from chain_gemm_fusion import BPE, MMA_MIN_M, STREAM_OVERHEAD, _dram_of, _eff_l2
from multi_gemm_fusion import analyze, segment_time

GPU = GPUS["h100-sxm"]
M = 131072
L = 6


def seg_detail(widths, M, gpu):
    """Replay segment_time's inner roofline for the OPTIMAL m0 and return full occupancy.

    Mirrors multi_gemm_fusion.segment_time / chain_gemm_fusion._roofline EXACTLY, but
    exposes out_tiles, sm_util, waves, active_sm, compute_eff, memory_time, bw_eff.
    Returns dict (or None if length-1 segment / infeasible)."""
    k = len(widths) - 1
    if k == 1:
        t, info = segment_time(widths, M, gpu)
        # replay a plain GEMM's occupancy from the estimator
        _, e = optimal_mapping_by_time(M, widths[1], widths[0], gpu, l2=True)
        return {"k": 1, "time": t, "m0": M, "mt": 1,
                "out_tiles": e.output_tiles, "sm_util": e.sm_utilization,
                "waves": e.waves, "active_sm": e.active_sm,
                "compute_eff": e.compute_time_eff_s, "memory": e.memory_time_s,
                "bw_eff": e.bw_eff_bytes_per_s, "bott": e.bottleneck,
                "num_stages": e.num_stages, "dram_mib": None}
    eff_l2 = _eff_l2(gpu)
    peak_pair = max(widths[s] + widths[s + 1] for s in range(k))
    m0_max = (gpu.smem_per_block_bytes - STREAM_OVERHEAD) // (peak_pair * BPE)
    if m0_max < MMA_MIN_M:
        return {"k": k, "time": float("inf"), "why": "INFEAS", "peak_pair": peak_pair,
                "m0_max": m0_max}
    ops = sum(2 * M * widths[s] * widths[s + 1] for s in range(k))
    best = None
    for m0 in divisors(M):
        if m0 < MMA_MIN_M or m0 > m0_max:
            continue
        mt = M // m0
        dram = 0.0
        last_bn = last_c = 1
        for s in range(k):
            w_in, w_out = widths[s], widths[s + 1]
            _, eb = optimal_mapping_by_time(m0, w_out, w_in, gpu, l2=True)
            w_big = (w_in * w_out * BPE) > eff_l2
            dram += (mt if w_big else 1) * _dram_of(eb, "W")
            if s == 0:
                dram += mt * _dram_of(eb, "A")
            if s == k - 1:
                dram += mt * _dram_of(eb, "OUT")
                last_bn, last_c = max(eb.mapping.bn, 1), eb.num_stages
        out_tiles = mt * max(1, widths[-1] // last_bn)
        resident = m0 * peak_pair * BPE + STREAM_OVERHEAD
        # ---- exact _roofline replica ----
        active_sm = min(max(out_tiles, 1), gpu.num_sm)
        waves = math.ceil(max(out_tiles, 1) / gpu.num_sm)
        sm_util = max(out_tiles, 1) / (waves * gpu.num_sm)
        per_sm_bw = gpu.bw_bytes_per_s / gpu.bw_saturation_sms
        bw_latency = active_sm * last_c * resident / gpu.latency_seconds
        bw_eff = min(gpu.bw_bytes_per_s, per_sm_bw * active_sm, bw_latency)
        compute_eff = (ops / gpu.peak_tensor_flops) / sm_util if sm_util > 0 else ops / gpu.peak_tensor_flops
        memory = dram / bw_eff if bw_eff > 0 else float("inf")
        t = max(compute_eff, memory)
        bott = "compute" if compute_eff >= memory else "memory"
        cand = {"k": k, "time": t, "m0": m0, "mt": mt, "out_tiles": out_tiles,
                "sm_util": sm_util, "waves": waves, "active_sm": active_sm,
                "compute_eff": compute_eff, "memory": memory, "bw_eff": bw_eff,
                "bott": bott, "num_stages": last_c, "last_bn": last_bn,
                "dram_mib": dram / 2**20}
        if best is None or t < best["time"]:
            best = cand
    return best


def fmt(x, nd=9):
    return "INF" if not math.isfinite(x) else f"{x*1e3:.{nd}f}"


for w in (512, 1024, 2048):
    widths = [w] * (L + 1)
    a = analyze(M, w, L, GPU)
    fa_t, fa_cuts, fa_segs = a["fuse_all"]
    uf_t, uf_cuts, uf_segs = a["unfused"]
    best_t, best_cuts, best_segs = a["best"]

    # best single-cut (2-segment) partition
    two_seg = [(t, c, s) for (t, c, s) in a["results"] if len(c) == 1 and math.isfinite(t)]
    two_seg.sort(key=lambda r: r[0])
    b2_t, b2_cuts, b2_segs = two_seg[0]

    print("=" * 92)
    print(f"w={w}  M={M}  L={L}  H100  | eff-L2={_eff_l2(GPU)/2**20:.0f}MiB  C_i=M*w*2={M*w*2/2**20:.0f}MiB")
    print(f"  fuse-all  cuts={fa_cuts or '()':<9}  t = {fmt(fa_t)} ms")
    print(f"  BEST      cuts={str(best_cuts):<9}  t = {fmt(best_t)} ms   (fuse_all/best = {fa_t/best_t:.9f})")
    print(f"  best 2seg cuts={str(b2_cuts):<9}  t = {fmt(b2_t)} ms   (fuse_all/2seg = {fa_t/b2_t:.9f})")
    print(f"  unfused   cuts={str(uf_cuts):<9}  t = {fmt(uf_t)} ms   (unfused/fuse_all = {uf_t/fa_t:.6f})")
    delta = fa_t - best_t
    print(f"  ABS diff fuse_all - best = {delta*1e3:.9f} ms  ({delta/fa_t*100:+.6f}% of fuse-all)")

    # occupancy: fuse-all's single segment
    fa_d = seg_detail(widths, M, GPU)
    print(f"  -- fuse-all single kernel (k=6) --")
    print(f"     m0={fa_d['m0']:>6} mt={fa_d['mt']:>4}  out_tiles={fa_d['out_tiles']:>7}  waves={fa_d['waves']:>4}  "
          f"active_sm={fa_d['active_sm']:>3}/{GPU.num_sm}  sm_util={fa_d['sm_util']:.6f}  bott={fa_d['bott']}")
    print(f"     compute_eff={fa_d['compute_eff']*1e3:.9f} ms  memory={fa_d['memory']*1e3:.9f} ms  "
          f"dram={fa_d['dram_mib']:.1f}MiB  last_bn={fa_d.get('last_bn')}")

    # occupancy: the best 2-segment split's two segments
    print(f"  -- best 2-seg split cuts={b2_cuts}: segments --")
    bounds = [0] + list(b2_cuts) + [L]
    seg_sum = 0.0
    for a0, b0 in zip(bounds, bounds[1:]):
        sd = seg_detail(widths[a0:b0 + 1], M, GPU)
        seg_sum += sd["time"]
        if sd["k"] == 1:
            print(f"     GEMMs {a0+1}-{b0} (k=1, plain): t={sd['time']*1e3:.9f} ms  out_tiles={sd['out_tiles']}  "
                  f"waves={sd['waves']}  active_sm={sd['active_sm']}  sm_util={sd['sm_util']:.6f}  "
                  f"compute_eff={sd['compute_eff']*1e3:.9f}  memory={sd['memory']*1e3:.9f}  bott={sd['bott']}")
        else:
            print(f"     GEMMs {a0+1}-{b0} (k={sd['k']}): t={sd['time']*1e3:.9f} ms  m0={sd['m0']} mt={sd['mt']}  "
                  f"out_tiles={sd['out_tiles']}  waves={sd['waves']}  active_sm={sd['active_sm']}  "
                  f"sm_util={sd['sm_util']:.6f}  compute_eff={sd['compute_eff']*1e3:.9f}  "
                  f"memory={sd['memory']*1e3:.9f}  bott={sd['bott']}")
    print(f"     2-seg total (sum of segs) = {seg_sum*1e3:.9f} ms")
    print()
