"""Rerun every fusion test with L2 REMOVED (l2_bytes=0), vs the L2-on baseline.

Removing L2 = every access goes to HBM, consistently for fused AND unfused: the inter-kernel
intermediate always round-trips HBM (no L2 to hold it), fused weight re-reads always hit HBM, and
each GEMM loses its cross-tile operand caching too. Implemented by zeroing the GPU's L2 capacity
(dataclasses.replace(g, l2_bytes=0)); the estimator then charges all snowcat traffic to DRAM, and
_eff_l2 -> 0 makes c_in_l2 always False and w_big always True in the models. No model code changes.
"""

from __future__ import annotations

import dataclasses
import itertools
import math

from gemm_time_estimator import GPUS
import chain_gemm_fusion as C
import chain_gemm_focused as F
from multi_gemm_fusion import analyze as m_analyze
from multi_gemm_smem import find_Nstar
from multi_gemm_smem import analyze as s_analyze

G = GPUS["h100-sxm"]
G0 = dataclasses.replace(G, name="H100 NO-L2", l2_bytes=0)


def count_chain(gpu):
    nf = nu = ni = 0
    for aa, ab, ad in itertools.product(C.ASPECTS, repeat=3):
        M, K1, N1, N2 = C.case_dims(aa, ab, ad)
        u, _ = C.unfused_time(M, K1, N1, N2, gpu)
        f, _ = C.fused_time(M, K1, N1, N2, gpu)
        if not math.isfinite(f):
            ni += 1
        elif f < u:
            nf += 1
        else:
            nu += 1
    return nf, nu, ni


def count_focused(gpu):
    nf = nu = ni = 0
    for N1, N2 in itertools.product(F.N1_VALUES, F.N2_VALUES):
        for K1 in F.K1_VALUES:
            u, _ = C.unfused_time(F.M, K1, N1, N2, gpu)
            f, _ = C.fused_time(F.M, K1, N1, N2, gpu)
            if not math.isfinite(f):
                ni += 1
            elif f < u:
                nf += 1
            else:
                nu += 1
    return nf, nu, ni


def hdr(t):
    print(f"\n{'='*72}\n{t}\n{'='*72}")


hdr("TEST 1 — chain 27-shape (fuse / unfuse / infeasible)")
for tag, gpu in [("L2-on ", G), ("NO-L2 ", G0)]:
    nf, nu, ni = count_chain(gpu)
    print(f"  {tag}: FUSE {nf:>2}   unfuse {nu:>2}   infeasible {ni:>2}")

hdr("TEST 2 — focused flash-attention regime, 24 cases (fuse / unfuse / infeasible)")
for tag, gpu in [("L2-on ", G), ("NO-L2 ", G0)]:
    nf, nu, ni = count_focused(gpu)
    print(f"  {tag}: FUSE {nf:>2}   unfuse {nu:>2}   infeasible {ni:>2}")

hdr("TEST 3 — multi-GEMM depth, uniform narrow (M=131072, L=6): fuse-all vs unfused")
print(f"  {'w':>5} {'':>2} {'fuse-all ms':>12} {'unfused ms':>11} {'speedup':>8} {'fuse-all opt?':>13}")
for w in [128, 256, 512, 1024]:
    for tag, gpu in [("L2", G), ("NL", G0)]:
        a = m_analyze(131072, w, 6, gpu)
        fa = a["fuse_all"][0]; uf = a["unfused"][0]; best = a["best"]
        opt = "YES" if (math.isfinite(fa) and best and best[1] == ()) else ("infeas" if not math.isfinite(fa) else "no")
        fam = f"{fa*1e3:.4f}" if math.isfinite(fa) else "INFEAS"
        sp = f"{uf/fa:.3f}x" if math.isfinite(fa) else "--"
        print(f"  {w:>5} {tag:>2} {fam:>12} {uf*1e3:>11.4f} {sp:>8} {opt:>13}")

hdr("TEST 4 — SMEM crossover N* (M=131072, seq schedule)")
print(f"  {'w':>5}   L2-on N*   NO-L2 N*")
for w in [512, 1024, 2048]:
    n_on, _ = find_Nstar(131072, w, G, "seq", 12)
    n_off, _ = find_Nstar(131072, w, G0, "seq", 12)
    print(f"  {w:>5}   {str(n_on):>7}   {str(n_off):>8}")

hdr("TEST 5 — SQUARE chain [n,n]@(n,n)xL, L=3 (the ties should move under NO-L2)")
print(f"  {'n':>5} {'':>2} {'C_i':>8} {'fuse-all ms':>12} {'unfused ms':>11} {'result':>18}")
for n in [1024, 2048, 4096]:
    for tag, gpu in [("L2", G), ("NL", G0)]:
        a = s_analyze(n, n, 3, gpu, "seq")
        fa = a["fuse_all"][0]; uf = a["unfused"][0]; best = a["best"]
        ci = n * n * C.BPE / 2**20
        if not math.isfinite(fa):
            res = "INFEASIBLE"
        elif abs(fa - uf) / uf < 1e-6:
            res = "tie"
        elif fa < uf:
            res = f"FUSE {uf/fa:.3f}x"
        else:
            res = f"unfuse {uf/fa:.3f}x"
        fam = f"{fa*1e3:.4f}" if math.isfinite(fa) else "INFEAS"
        print(f"  {n:>5} {tag:>2} {ci:>6.0f}Mi {fam:>12} {uf*1e3:>11.4f} {res:>18}")

print("\n(NO-L2 = H100 with l2_bytes=0; every access to HBM, fused & unfused alike.)")
