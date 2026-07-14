#!/usr/bin/env python
"""ANALYTICAL comparison of the two estimators on an H100 SXM5 spec profile.

Nothing runs on a GPU. For each GEMM shape we report:
  * each model's --optimal mapping (min estimated time, BM,BN>=64, BK>=32, split_k=1)
  * both models evaluated on a common set of fixed tiles -> snow/min divergence
  * bottleneck labels, to show WHERE the traffic model starts to matter.

Run:  PYTHONPATH=/home/shuhan/GEMM-mapping:/home/shuhan/snowcat-demo \
        conda run -n profiling python h100_analytical_compare.py
"""
from __future__ import annotations

import math
import sys

sys.path.insert(0, "/home/shuhan/GEMM-mapping")
sys.path.append("/home/shuhan/snowcat-demo")

from gemm_time_estimator import (  # noqa: E402
    GPUS, Mapping, estimate_best_order, optimal_mapping_by_time)
from gemm_time_estimator_min import (  # noqa: E402
    estimate_gemm_time_min, optimal_mapping_min_by_time)

GPU = GPUS["h100-sxm"]
RIDGE = GPU.peak_tensor_flops / GPU.bw_bytes_per_s

SHAPES = [
    ("square-small",   512,   512,   512),
    ("square-mid",     2048,  2048,  2048),
    ("square-4k",      4096,  4096,  4096),
    ("square-8k",      8192,  8192,  8192),
    ("square-16k",     16384, 16384, 16384),
    ("skinny-M128",    128,   4096,  4096),
    ("skinny-M128-XL", 128,   16384, 16384),
    ("skinny-N128",    4096,  128,   4096),
    ("short-K256",     4096,  4096,  256),
    ("deep-K8192",     512,   512,   8192),
    ("wide",           1024,  8192,  2048),
    ("tall",           8192,  1024,  2048),
    ("llm-ffn-up",     2048,  28672, 8192),
    ("llm-ffn-down",   2048,  8192,  28672),
    ("decode-M32",     32,    8192,  8192),
]

FIXED_TILES = [(64, 64, 32), (128, 128, 32), (128, 128, 64),
               (128, 256, 32), (256, 128, 32)]


def snow_est(m, n, k, bm, bn, bk):
    mp = Mapping(bm=bm, bn=bn, bk=bk, loop_order=("M", "N", "K"),
                 num_stages=None, split_k=1)
    e, order = estimate_best_order(m, n, k, mp, GPU)
    return e, order


def min_est(m, n, k, bm, bn, bk):
    mp = Mapping(bm=bm, bn=bn, bk=bk, loop_order=("M", "N", "K"),
                 num_stages=None, split_k=1)
    return estimate_gemm_time_min(m, n, k, mp, GPU)


def main():
    print(f"GPU: {GPU.name}")
    print(f"  compute roof {GPU.peak_tensor_flops/1e12:.0f} TFLOP/s, "
          f"BW {GPU.bw_bytes_per_s/1e12:.2f} TB/s, ridge OI = {RIDGE:.0f} FLOP/B "
          f"(4060-locked ridge was ~108)\n")

    diverge = []
    for pattern, m, n, k in SHAPES:
        oi_floor = 2*m*n*k / ((m*k + k*n + m*n) * GPU.bytes_per_element)
        print(f"=== {pattern}: {m}x{n}x{k}   (OI at traffic floor = {oi_floor:.0f}, "
              f"{'compute' if oi_floor > RIDGE else 'memory'}-side of ridge) ===")

        smap, se = optimal_mapping_by_time(m, n, k, GPU)
        mmap, me = optimal_mapping_min_by_time(m, n, k, GPU)
        r_opt = se.time_s / me.time_s
        print(f"  snow --optimal: {smap.bm:>4}x{smap.bn:<4}x{smap.bk:<3} "
              f"order={'-'.join(smap.loop_order):<5} {se.time_s*1e3:9.4f} ms "
              f"[{se.bottleneck}]")
        print(f"  min  --optimal: {mmap.bm:>4}x{mmap.bn:<4}x{mmap.bk:<3} "
              f"{'':<11} {me.time_s*1e3:9.4f} ms [{me.bottleneck}]"
              f"   snow/min(opt) = {r_opt:.2f}x")

        rows = []
        for bm, bn, bk in FIXED_TILES:
            if m % bm or n % bn or k % bk:
                continue
            try:
                es, order = snow_est(m, n, k, bm, bn, bk)
                em = min_est(m, n, k, bm, bn, bk)
            except ValueError:
                continue
            if not (es.fits_smem and em.fits_smem):
                continue
            r = es.time_s / em.time_s
            rows.append((bm, bn, bk, es, em, r, order))
            diverge.append((pattern, f"{bm}x{bn}x{bk}", r,
                            es.bottleneck, em.bottleneck))
        if rows:
            print(f"    {'tile':<12} {'snow ms':>10} {'bound':<8} {'min ms':>10} "
                  f"{'bound':<8} {'snow/min':>8}")
            for bm, bn, bk, es, em, r, order in rows:
                mark = "  <== DIVERGE" if r > 1.05 else ""
                print(f"    {f'{bm}x{bn}x{bk}':<12} {es.time_s*1e3:>10.4f} "
                      f"{es.bottleneck:<8} {em.time_s*1e3:>10.4f} {em.bottleneck:<8} "
                      f"{r:>7.2f}x{mark}")
        print()

    n_all = len(diverge)
    n_div = sum(1 for *_, r, _, _ in [(0, 0, r, a, b) for _, _, r, a, b in diverge] if r > 1.05)
    n_div = sum(1 for _, _, r, _, _ in diverge if r > 1.05)
    worst = max(diverge, key=lambda d: d[2])
    print("================== SUMMARY ==================")
    print(f"(shape,tile) points evaluated : {n_all}")
    print(f"points where models diverge >5%: {n_div}  ({100*n_div/n_all:.0f}%)")
    print(f"largest divergence            : {worst[2]:.2f}x  at {worst[0]} tile {worst[1]} "
          f"(snow {worst[3]}-bound, min {worst[4]}-bound)")
    bb = sum(1 for _, _, r, a, b in diverge if a != b)
    print(f"points with different bottleneck labels: {bb}")


if __name__ == "__main__":
    main()
