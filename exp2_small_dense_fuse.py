"""EXPERIMENT 2 — is the small/dense-model fusion win REAL (realizable) on C500, not just a ceiling?

Parts:
 (a) Small dense FFN (H=1024, M=2048): measure torch.compile FUSED up_gate+SwiGLU vs unfused eager.
     Does it capture a chunk of the 28% ceiling? (GLM's compile captured ~0.)  Also H=512,2048 for trend.
 (b) Ridge/dim law extra points: H in {512,1536,3072} dense-FFN ceiling (+ re-confirm 1024,2048).
 (c) Is the H=1024 GEMM really MEMORY-bound (103 TF/s << 226 peak)? Compute true roofline:
     arithmetic intensity (FLOP/byte) vs ridge 158, and achieved HBM BW vs 1.43 TB/s peak.

Fusion up_gate: GEMM [M,2H,K=H] -> split gate/up [M,H] each -> silu(gate)*up -> [M,H].
Ceiling = swiglu/(gemm+swiglu). Realized = (unfused_eager - fused_compiled)/unfused_eager.
"""
import json, statistics, time
import torch
import torch.nn.functional as F

DEV = "cuda:0"
PEAK_TF = 226.0        # measured C500 bf16 TFLOP/s
PEAK_BW = 1.43e12      # C500 HBM B/s
RIDGE = 158.0          # ridge-point OI (FLOP/byte)
M = 2048
BPE = 2                # bytes / bf16 elem


def med(fn, iters=30, warmup=12):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    return statistics.median(ts)


def bf(*s): return torch.randn(*s, device=DEV, dtype=torch.bfloat16)


def gemm_stats(m, n, k, t):
    flops = 2 * m * n * k
    hbm_bytes = BPE * (m * k + k * n + m * n)          # read A,B ; write C (minimal HBM traffic)
    return {"tflops": flops / t / 1e12,
            "oi_flop_per_byte": flops / hbm_bytes,
            "achieved_bw_GBs": hbm_bytes / t / 1e9,
            "bw_frac_of_peak": hbm_bytes / t / PEAK_BW}


out = {"partB_ceiling": [], "partA_fused": [], "partC_roofline": {}}

# ---------- Part B + C: ceiling + roofline across H ----------
print("# Part B/C: dense FFN up_gate+SwiGLU ceiling & GEMM roofline (M=2048, inter=H)")
print(f"{'H':>5} {'gemm_us':>8} {'TFLOP/s':>8} {'OI':>7} {'BW_GB/s':>8} {'%peakBW':>7} "
      f"{'swig_us':>8} {'ceil%':>6} {'roofline':>9} {'heur':>6}")
for H in [512, 1024, 1536, 2048, 3072, 4096]:
    N = 2 * H
    a, b = bf(M, H), bf(H, N)
    g = med(lambda: a @ b, iters=25, warmup=10)
    del a, b; torch.cuda.empty_cache()
    st = gemm_stats(M, N, H, g)
    gg, uu = bf(M, H), bf(M, H)
    v = med(lambda: F.silu(gg) * uu, iters=40, warmup=15)
    del gg, uu; torch.cuda.empty_cache()
    ceil = v / (g + v) * 100
    roofline = "compute" if st["oi_flop_per_byte"] > RIDGE else "memory"   # TRUE roofline
    heur = "compute" if st["tflops"] > 0.7 * PEAK_TF else "memory"          # old throughput heuristic
    rec = {"hidden": H, "gemm_ms": g, "swiglu_ms": v, "ceiling_pct": ceil,
           "roofline_bound": roofline, "heuristic_bound": heur, **st}
    out["partB_ceiling"].append(rec)
    print(f"{H:>5} {g*1e6:8.1f} {st['tflops']:8.1f} {st['oi_flop_per_byte']:7.0f} "
          f"{st['achieved_bw_GBs']:8.0f} {st['bw_frac_of_peak']*100:7.1f} {v*1e6:8.1f} "
          f"{ceil:6.1f} {roofline:>9} {heur:>6}")

out["partC_roofline"] = next(r for r in out["partB_ceiling"] if r["hidden"] == 1024)

# ---------- Part A: REAL fused measurement via torch.compile ----------
print("\n# Part A: torch.compile fused up_gate+SwiGLU vs unfused eager")
print(f"{'H':>5} {'unfused_us':>10} {'gemm_us':>8} {'ideal_us':>8} {'compiled_us':>11} "
      f"{'ceil%':>6} {'realized%':>9} {'frac_of_ceil':>12} {'compile':>10}")

def make_fns(H):
    Wshape_n = 2 * H
    x = bf(M, H)
    W = bf(H, Wshape_n)
    def up_swiglu(x, W):
        gu = x @ W
        g, u = gu[:, :H], gu[:, H:]
        return F.silu(g) * u
    return x, W, up_swiglu

for H in [512, 1024, 2048]:
    x, W, fn = make_fns(H)
    # pure gemm (ideal fused lower bound = just the matmul time)
    gemm_only = med(lambda: x @ W, iters=25, warmup=10)
    # unfused eager: matmul + slice + silu + mul (multiple kernels)
    unfused = med(lambda: fn(x, W), iters=25, warmup=10)
    rec = {"hidden": H, "gemm_only_ms": gemm_only, "unfused_ms": unfused}
    # default torch.compile
    comp_status = "ok"
    compiled_ms = None
    try:
        cfn = torch.compile(fn)
        # extra warmup for compilation
        for _ in range(6): cfn(x, W)
        torch.cuda.synchronize()
        compiled_ms = med(lambda: cfn(x, W), iters=25, warmup=10)
    except Exception as ex:
        comp_status = "ERR:" + str(ex)[:80]
    # max-autotune (best shot at template epilogue fusion)
    mautotune_ms = None
    ma_status = "ok"
    try:
        cfn2 = torch.compile(fn, mode="max-autotune")
        for _ in range(6): cfn2(x, W)
        torch.cuda.synchronize()
        mautotune_ms = med(lambda: cfn2(x, W), iters=25, warmup=10)
    except Exception as ex:
        ma_status = "ERR:" + str(ex)[:80]
    # correctness check
    with torch.no_grad():
        ref = fn(x, W)
        ok = True
        if compiled_ms is not None:
            try: ok = torch.allclose(cfn(x, W), ref, atol=1e-2, rtol=1e-2)
            except Exception: ok = False

    swig_ms = unfused - gemm_only   # approx vec contribution actually paid in eager
    ceil = swig_ms / unfused * 100 if unfused > 0 else 0
    best_fused = min([t for t in [compiled_ms, mautotune_ms] if t is not None], default=None)
    realized = (unfused - best_fused) / unfused * 100 if best_fused else None
    frac = (realized / ceil * 100) if (realized is not None and ceil > 0) else None
    rec.update({"compiled_ms": compiled_ms, "maxautotune_ms": mautotune_ms,
                "compile_status": comp_status, "maxautotune_status": ma_status,
                "ceiling_pct": ceil, "realized_pct": realized,
                "frac_of_ceiling_pct": frac, "correct": ok})
    out["partA_fused"].append(rec)
    cm = f"{compiled_ms*1e6:.1f}" if compiled_ms else "ERR"
    mm = f"{mautotune_ms*1e6:.1f}" if mautotune_ms else "ERR"
    rp = f"{realized:.1f}" if realized is not None else "NA"
    fc = f"{frac:.0f}" if frac is not None else "NA"
    print(f"{H:>5} {unfused*1e6:10.1f} {gemm_only*1e6:8.1f} {gemm_only*1e6:8.1f} "
          f"{cm:>11} {ceil:6.1f} {rp:>9} {fc:>12} default={cm} ma={mm} ok={ok}")

with open("exp2_small_dense_fuse.json", "w") as f:
    json.dump(out, f, indent=1)
print("\n[wrote exp2_small_dense_fuse.json]")
