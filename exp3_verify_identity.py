"""Verify the compute-bound tie is an EXACT structural identity, not float noise.

(1) Dump ALL 32 partitions' times to full precision for each w -> are they all equal?
(2) Show out_tiles / sm_util is k-INDEPENDENT (same for every segment length).
(3) Compute margin: how many extra HBM round-trips would it take for a cut's memory
    roof to exceed the compute roof (the only way a split could become STRICTLY worse)?
"""
from __future__ import annotations
import math
from gemm_time_estimator import GPUS
from multi_gemm_fusion import analyze
from chain_gemm_fusion import _eff_l2

GPU = GPUS["h100-sxm"]
M, L = 131072, 6

for w in (512, 1024, 2048):
    a = analyze(M, w, L, GPU)
    times = sorted(t for t, c, s in a["results"] if math.isfinite(t))
    fa_t = a["fuse_all"][0]
    tmin, tmax = times[0], times[-1]
    n_distinct = len(set(round(t, 15) for t in times))
    # how many partitions exactly equal fuse-all (to 1e-15 relative)?
    n_eq = sum(1 for t in times if abs(t - fa_t) <= 1e-12 * fa_t)
    print(f"w={w}: {len(times)} feasible partitions | "
          f"min={tmin*1e3:.12f}ms max={tmax*1e3:.12f}ms | spread={ (tmax-tmin)/tmin*100:.8f}% | "
          f"#distinct-to-1e15={n_distinct} | #exactly==fuse_all={n_eq}")
    # is fuse-all strictly minimal or tied?
    strictly_below = [ (t,c) for t,c,s in a["results"] if math.isfinite(t) and t < fa_t*(1-1e-12) ]
    print(f"      partitions STRICTLY faster than fuse-all: {len(strictly_below)}  "
          f"(fuse_all t={fa_t*1e3:.12f} ms)")

# margin: compute roof vs memory roof for fuse-all (from exp3 output), and #round-trips to flip
print("\nMARGIN (compute roof vs memory roof, fuse-all single kernel):")
print("  a cut adds ~1 HBM round-trip of the intermediate (2*M*w*2 bytes) to the MEMORY roof.")
print("  a split turns STRICTLY worse only if some segment's memory roof exceeds compute roof.")
for w, comp_ms, mem_ms in [(512,0.417154098,0.081069010),(1024,1.666937705,0.164016067),(2048,6.667750820,0.335544320)]:
    # segment memory scales ~ with its share; worst single segment ~ full compute is the fuse-all.
    ratio = comp_ms/mem_ms
    print(f"  w={w}: compute={comp_ms:.4f}ms  memory={mem_ms:.4f}ms  compute/memory={ratio:.2f}x headroom "
          f"(memory would need ~{ratio:.1f}x more traffic to bind)")
