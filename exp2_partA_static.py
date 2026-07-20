"""Part A, fair version: static-shape torch.compile (dynamic=False, dynamo.reset per H) so compile
gets its best shot at fusing up_gate GEMM + SwiGLU. If it STILL cannot beat unfused eager, the
Triton-template SMEM blocker (96KB > C500's 64KB) means epilogue fusion needs a custom kernel."""
import json, statistics
import torch
import torch.nn.functional as F
import torch._dynamo as dynamo

DEV = "cuda:0"; M = 2048


def med(fn, iters=30, warmup=15):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    return statistics.median(ts)


def bf(*s): return torch.randn(*s, device=DEV, dtype=torch.bfloat16)

out = []
print(f"{'H':>5} {'gemm_us':>8} {'unfused_us':>10} {'compiled_us':>11} {'ma_us':>8} "
      f"{'ceil%':>6} {'realized%':>9} {'best_beats_eager':>16}")
for H in [512, 1024, 2048]:
    N = 2 * H
    x, W = bf(M, H), bf(H, N)
    def up_swiglu(x, W):
        gu = x @ W
        g, u = gu[:, :H], gu[:, H:]
        return F.silu(g) * u
    gemm_only = med(lambda: x @ W, iters=25, warmup=10)
    unfused = med(lambda: up_swiglu(x, W), iters=25, warmup=10)

    dynamo.reset()
    cfn = torch.compile(up_swiglu, dynamic=False)
    for _ in range(8): cfn(x, W)
    torch.cuda.synchronize()
    compiled = med(lambda: cfn(x, W), iters=25, warmup=10)

    dynamo.reset()
    ma_ms = None
    try:
        cma = torch.compile(up_swiglu, dynamic=False, mode="max-autotune")
        for _ in range(8): cma(x, W)
        torch.cuda.synchronize()
        ma_ms = med(lambda: cma(x, W), iters=25, warmup=10)
    except Exception as ex:
        ma_ms = None
    ok = bool(torch.allclose(cfn(x, W), up_swiglu(x, W), atol=1e-2, rtol=1e-2))

    best = min([t for t in [compiled, ma_ms] if t is not None])
    ceil = (unfused - gemm_only) / unfused * 100
    realized = (unfused - best) / unfused * 100
    beats = best < unfused
    out.append({"hidden": H, "gemm_only_ms": gemm_only, "unfused_ms": unfused,
                "compiled_ms": compiled, "maxautotune_ms": ma_ms, "ceiling_pct": ceil,
                "realized_pct": realized, "best_beats_eager": beats, "correct": ok})
    mm = f"{ma_ms*1e6:.1f}" if ma_ms else "NA"
    print(f"{H:>5} {gemm_only*1e6:8.1f} {unfused*1e6:10.1f} {compiled*1e6:11.1f} {mm:>8} "
          f"{ceil:6.1f} {realized:9.1f} {str(beats):>16}")
    del x, W; torch.cuda.empty_cache()

with open("exp2_partA_static.json", "w") as f:
    json.dump(out, f, indent=1)
print("[wrote exp2_partA_static.json]")
