"""Actually IMPLEMENT the fusion via Triton GEMM templates (torch.compile max-autotune) on C500.

Default torch.compile keeps the vendor GEMM opaque (no epilogue fusion). max-autotune emits a Triton
GEMM template that CAN fuse the SwiGLU/residual epilogue into the GEMM (intermediate stays on-chip).
This is the real fusion algorithm. Compare eager (unfused) vs max-autotune-fused; verify correctness.
"""
import statistics
import torch
import torch.nn.functional as F
import torch._inductor.config as ic

ic.max_autotune = True
ic.max_autotune_gemm = True
DEV = "cuda:0"
INT = 2048


def med(fn, iters=25, warmup=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    return statistics.median(ts)


def bf(*s): return torch.randn(*s, device=DEV, dtype=torch.bfloat16)


def compare(name, fn, args, close_rtol=2e-2):
    eager = med(lambda: fn(*args))
    cf = torch.compile(fn, mode="max-autotune")
    # correctness + warmup compile
    o_e, o_c = fn(*args), cf(*args)
    torch.cuda.synchronize()
    ok = torch.allclose(o_e.float(), o_c.float(), rtol=close_rtol, atol=1e-1)
    fused = med(lambda: cf(*args))
    sp = eager / fused
    print(f"{name:>26}: eager {eager*1e3:8.4f}  fused(max-autotune) {fused*1e3:8.4f}  "
          f"speedup {sp:5.3f}x  {'OK' if ok else 'MISMATCH'}")
    return sp


# ---- F1: mla_o + residual (dense mm) ----
a_o, b_o, res = bf(2048, 16384), bf(16384, 6144), bf(2048, 6144)
compare("F1 mla_o+residual", lambda a, b, r: a @ b + r, (a_o, b_o, res))

# ---- F4 dense: up_gate + SwiGLU, sweep hidden ----
def f4(x, W):
    gu = x @ W
    g, u = gu[:, :gu.shape[1] // 2], gu[:, gu.shape[1] // 2:]
    return F.silu(g) * u
for H in [2048, 4096, 6144]:      # 6144 = GLM hidden (dense variant)
    x, W = bf(2048, H), bf(H, 2 * H)
    compare(f"F4 dense up_gate+swiglu H={H}", f4, (x, W))

# ---- F4 small dense (high-ceiling regime) ----
for H in [1024, 2048]:
    x, W = bf(2048, H), bf(H, 2 * H)
    compare(f"F4 small-dense H={H}", f4, (x, W))

# ---- F4 GLM MoE (grouped bmm) — does max-autotune fuse a batched GEMM? ----
def f4moe(x, W):
    gu = torch.bmm(x, W)
    g, u = gu[..., :INT], gu[..., INT:]
    return F.silu(g) * u
xu, Wu = bf(256, 64, 6144), bf(256, 6144, 2 * INT)
compare("F4 GLM MoE (bmm+swiglu)", f4moe, (xu, Wu))

# ---- F5 GLM MoE: SwiGLU + down (grouped) ----
def f5moe(g, u, W):
    return torch.bmm(F.silu(g) * u, W)
g5, u5, Wd = bf(256, 64, INT), bf(256, 64, INT), bf(256, INT, 6144)
compare("F5 GLM MoE (swiglu+down)", f5moe, (g5, u5, Wd))
