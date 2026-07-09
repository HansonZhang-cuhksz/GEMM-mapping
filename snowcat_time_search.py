#!/usr/bin/env python
"""Snowcat optimal-mapping search by TWO objectives, non-split-K:
   (a) minimum traffic  -- what the current --optimal returns
   (b) minimum estimated TIME -- using the occupancy-calibrated model
and compare the winner to CUTLASS's non-split-K auto pick (64x64x32).
"""
import sys
from snowcat_demo.model.mapping import enumerate_mappings
from snowcat_demo.model.workload import GemmWorkload
from gemm_time_estimator import estimate_gemm_time, Mapping, GPUS

M, N, K = 128, 4096, 4096
gpu = GPUS["rtx4060-measured"]
MMA_MIN_BM, MMA_MIN_BN, MMA_MIN_BK = 16, 8, 16

wl = GemmWorkload(m=M, k=K, n=N, bytes_per_element=gpu.bytes_per_element)
rows = []
for p in enumerate_mappings(wl):
    mp = p.mapping
    if mp.m0 < MMA_MIN_BM or mp.n0 < MMA_MIN_BN or mp.k0 < MMA_MIN_BK:
        continue
    m = Mapping(bm=mp.m0, bn=mp.n0, bk=mp.k0, loop_order=mp.loop_order,
                num_stages=None, split_k=1)
    try:
        e = estimate_gemm_time(M, N, K, m, gpu)
    except ValueError:
        continue
    if not e.fits_smem:
        continue
    rows.append((mp.m0, mp.n0, mp.k0, "-".join(mp.loop_order),
                 e.traffic_bytes, e.time_s, e.bottleneck))

def show(title, key, n=6, only_bk_ge_32=False):
    pool = [r for r in rows if (r[2] >= 32 or not only_bk_ge_32)]
    pool.sort(key=key)
    print(title)
    print(f"  {'BM':>4}x{'BN':<4} {'BK':>3}  {'order':<6} {'T(MiB)':>8} {'time(ms)':>9}  {'bound':<8}")
    for m0,n0,k0,order,T,t,b in pool[:n]:
        print(f"  {m0:>4}x{n0:<4} {k0:>3}  {order:<6} {T/2**20:>8.2f} {t*1e3:>9.4f}  {b:<8}")
    return pool[0]

print(f"GEMM {M}x{N}x{K}  (non-split-K)   candidates evaluated: {len(rows)}\n")
best_traffic = show("[A] snowcat --optimal  == minimize TRAFFIC:", key=lambda r: r[4])
print()
best_time_any = show("[B] minimize TIME (corrected model), any BK:", key=lambda r: r[5])
print()
best_time_bk32 = show("[C] minimize TIME, BK>=32 (CUTLASS-runnable):", key=lambda r: r[5], only_bk_ge_32=True)

print()
print(f"traffic-optimal tile : {best_traffic[0]}x{best_traffic[1]}x{best_traffic[2]}  ({best_traffic[4]/2**20:.1f} MiB, {best_traffic[5]*1e3:.3f} ms)")
print(f"time-optimal    tile : {best_time_any[0]}x{best_time_any[1]}x{best_time_any[2]}  ({best_time_any[5]*1e3:.3f} ms)")
print(f"time-optimal BK>=32  : {best_time_bk32[0]}x{best_time_bk32[1]}x{best_time_bk32[2]}  ({best_time_bk32[5]*1e3:.3f} ms)")
print(f"CUTLASS --no-splitk auto pick (measured): 64x64x32  (~0.31 ms)")
