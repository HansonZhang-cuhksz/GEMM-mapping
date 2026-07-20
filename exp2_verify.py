"""EXPERIMENT 2 - verify the two surprises on the physical MetaX C500 (fusion env, bf16, cuda:0).

(a) Is torch.addmm the cuBLAS FUSED beta-accumulate path or a fallback mm+add? Numerical + kernel
    count + re-time addmm vs mm vs mm+add at mla_o shape. baddbmm too.
(b) Did torch.compile fuse F4 bmm+swiglu? Backend compiles at all? Graph breaks? Hand-fused F4 traffic.
(c) Grouped FFN weight-bandwidth-bound? effective BW = weight_bytes/time vs 1.43 TB/s peak.

Prints clearly-labelled sections; saves exp2_verify.json.
"""
import json, statistics
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

DEV = "cuda:0"
HIDDEN, INTERMEDIATE, EXPERTS = 6144, 2048, 256
BATCH, TOP_K = 2048, 8
TPE = BATCH * TOP_K // EXPERTS   # 64
KV = 64 * 256                    # 16384
PEAK_BW = 1.43e12                # measured C500 HBM peak, B/s


def bf(*shape):
    return torch.randn(*shape, device=DEV, dtype=torch.bfloat16)


def med_time(fn, iters=40, warmup=12):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    return statistics.median(ts)


def count_kernels(fn, reps=5):
    """Return (n_device_kernels_per_call, list of (name, us)) via CUDA profiler, averaged over reps."""
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    kevs = [e for e in prof.events() if getattr(e, "device_time", 0) and e.device_time > 0]
    # aggregate by name
    agg = {}
    for e in kevs:
        agg.setdefault(e.key, [0, 0.0])
        agg[e.key][0] += e.count if e.count else 1
        agg[e.key][1] += e.device_time
    total_launches = sum(v[0] for v in agg.values())
    per_call = total_launches / reps
    names = [(k, round(v[1] / reps, 2)) for k, v in sorted(agg.items(), key=lambda kv: -kv[1][1])]
    return per_call, names


out = {}
print("=" * 78)
print("DEVICE:", torch.cuda.get_device_name(0), "| torch", torch.__version__,
      "| n_gpu", torch.cuda.device_count())
p = torch.cuda.get_device_properties(0)
print(f"SM {p.multi_processor_count} | L2 {p.L2_cache_size/2**20:.0f} MiB | "
      f"total mem {p.total_memory/2**30:.1f} GiB")

# =====================================================================================
# (a) addmm: fused cuBLAS beta-accumulate vs fallback mm+add
# =====================================================================================
print("\n" + "=" * 78)
print("(a) torch.addmm at mla_o shape  A[2048,16384] @ B[16384,6144] + res[2048,6144]")
print("=" * 78)
a_o, b_o, res = bf(BATCH, KV), bf(KV, HIDDEN), bf(BATCH, HIDDEN)

r_add = torch.addmm(res, a_o, b_o)
r_ref = res + a_o @ b_o
diff = (r_add.float() - r_ref.float()).abs()
rel = diff / (r_ref.float().abs() + 1e-3)
print(f"numerical: addmm == res+a@b  max_abs_diff={diff.max().item():.4g}  "
      f"max_rel={rel.max().item():.4g}  mean_abs={diff.mean().item():.4g}")
out["a_addmm_max_abs_diff"] = diff.max().item()
out["a_addmm_max_rel_diff"] = rel.max().item()

k_mm, names_mm = count_kernels(lambda: a_o @ b_o)
k_addmm, names_addmm = count_kernels(lambda: torch.addmm(res, a_o, b_o))
k_mmadd, names_mmadd = count_kernels(lambda: (a_o @ b_o) + res)
print(f"\nkernel launches / call:  mm={k_mm:.1f}   addmm={k_addmm:.1f}   mm+add={k_mmadd:.1f}")
print("  mm     kernels:", names_mm)
print("  addmm  kernels:", names_addmm)
print("  mm+add kernels:", names_mmadd)
out["a_kernels_mm"] = k_mm
out["a_kernels_addmm"] = k_addmm
out["a_kernels_mm_add"] = k_mmadd

t_mm = med_time(lambda: a_o @ b_o, iters=30, warmup=10)
t_addmm = med_time(lambda: torch.addmm(res, a_o, b_o), iters=30, warmup=10)
t_mmadd = med_time(lambda: (a_o @ b_o) + res, iters=30, warmup=10)
print(f"\ntimes (ms):  mm={t_mm*1e3:.4f}   mm+add={t_mmadd*1e3:.4f}   addmm(fused)={t_addmm*1e3:.4f}")
print(f"  addmm vs mm+add   : {t_mmadd/t_addmm:.4f}x  ({'addmm faster' if t_addmm<t_mmadd else 'addmm slower'})")
print(f"  addmm vs mm(no add): {t_addmm/t_mm:.4f}x  (epilogue-add overhead over bare GEMM)")
print(f"  mm+add vs mm       : {t_mmadd/t_mm:.4f}x  (separate-add-kernel overhead over bare GEMM)")
out["a_t_mm"] = t_mm; out["a_t_addmm"] = t_addmm; out["a_t_mm_add"] = t_mmadd

# baddbmm (batched beta-accumulate) at a grouped shape
xg = bf(EXPERTS, TPE, HIDDEN)
Wg = bf(EXPERTS, HIDDEN, 2 * INTERMEDIATE)
cg = bf(EXPERTS, TPE, 2 * INTERMEDIATE)
r_badd = torch.baddbmm(cg, xg, Wg)
r_bref = cg + torch.bmm(xg, Wg)
bdiff = (r_badd.float() - r_bref.float()).abs().max().item()
k_bmm, _ = count_kernels(lambda: torch.bmm(xg, Wg))
k_baddbmm, nb = count_kernels(lambda: torch.baddbmm(cg, xg, Wg))
k_bmmadd, _ = count_kernels(lambda: torch.bmm(xg, Wg) + cg)
t_bmm = med_time(lambda: torch.bmm(xg, Wg), iters=20, warmup=8)
t_baddbmm = med_time(lambda: torch.baddbmm(cg, xg, Wg), iters=20, warmup=8)
t_bmmadd = med_time(lambda: torch.bmm(xg, Wg) + cg, iters=20, warmup=8)
print(f"\nbaddbmm (grouped 256x64x6144x4096, C-accumulate):  max_abs_diff={bdiff:.4g}")
print(f"  kernels: bmm={k_bmm:.1f}  baddbmm={k_baddbmm:.1f}  bmm+add={k_bmmadd:.1f}")
print(f"  times(ms): bmm={t_bmm*1e3:.4f}  bmm+add={t_bmmadd*1e3:.4f}  baddbmm(fused)={t_baddbmm*1e3:.4f}"
      f"  -> baddbmm vs bmm+add {t_bmmadd/t_baddbmm:.4f}x")
out["a_baddbmm_max_abs_diff"] = bdiff
out["a_kernels_baddbmm"] = k_baddbmm
out["a_t_bmm"] = t_bmm; out["a_t_baddbmm"] = t_baddbmm; out["a_t_bmm_add"] = t_bmmadd

with open("exp2_verify.json", "w") as f:
    json.dump(out, f, indent=1)
print("\n[wrote exp2_verify.json]")
