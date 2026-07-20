"""Measure the GLM-5.2 decode MoE kernels + a real cuBLAS-fused F1 on the physical C500 (fusion env).

Uses the cuBLAS counterpart via torch: addmm = residual + x@W in ONE GEMM call (beta=1 accumulate) =
exactly F1's fused mla_o+residual. Grouped MoE GEMMs via bmm (256 experts). Saves metax_glm.json.
"""
import json, statistics
import torch
import torch.nn.functional as F

DEV = "cuda:0"
BPE = 2
HIDDEN, INTERMEDIATE, EXPERTS = 6144, 2048, 256
BATCH, TOP_K = 2048, 8
TPE = BATCH * TOP_K // EXPERTS          # 64 tokens/expert
KV = 64 * 256                            # N_HEADS*V_HEAD_DIM = 16384


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


def bf(*shape):
    return torch.randn(*shape, device=DEV, dtype=torch.bfloat16)


out = {}

# ---- component GEMMs (unfused) ----
a_o, b_o = bf(BATCH, KV), bf(KV, HIDDEN)                       # mla_o 2048x6144x16384
out["mla_o"] = med_time(lambda: a_o @ b_o, iters=20, warmup=8)
a_r, b_r = bf(BATCH, HIDDEN), bf(HIDDEN, 256)                  # router 2048x256x6144
out["router"] = med_time(lambda: a_r @ b_r)
xu, Wu = bf(EXPERTS, TPE, HIDDEN), bf(EXPERTS, HIDDEN, 2 * INTERMEDIATE)   # up_gate grouped
out["up_gate_grouped"] = med_time(lambda: torch.bmm(xu, Wu), iters=20, warmup=8)
xd, Wd = bf(EXPERTS, TPE, INTERMEDIATE), bf(EXPERTS, INTERMEDIATE, HIDDEN)  # down grouped
out["down_grouped"] = med_time(lambda: torch.bmm(xd, Wd), iters=20, warmup=8)

# ---- vector kernels ----
y1, y2 = bf(BATCH, HIDDEN), bf(BATCH, HIDDEN)
out["residual"] = med_time(lambda: y1 + y2)
w_rms = bf(HIDDEN)
def rmsnorm():
    h = y1.float()
    return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16) * w_rms
out["rmsnorm"] = med_time(rmsnorm)
gate, up = bf(BATCH * TOP_K, INTERMEDIATE), bf(BATCH * TOP_K, INTERMEDIATE)   # 16384 x 2048 each
out["swiglu"] = med_time(lambda: F.silu(gate) * up)

# ---- F1 fusion: real cuBLAS beta-accumulate (residual + mla_o in ONE GEMM) ----
res = bf(BATCH, HIDDEN)
# unfused: GEMM then add (two kernels); fused: addmm (one GEMM with C-accumulate)
out["F1_unfused_mm_then_add"] = med_time(lambda: (a_o @ b_o) + res, iters=20, warmup=8)
out["F1_fused_addmm"] = med_time(lambda: torch.addmm(res, a_o, b_o), iters=20, warmup=8)

# ---- F4 attempt: fuse up_gate GEMM + SwiGLU via torch.compile (writes half-width activated) ----
try:
    def swiglu_from_gemm(x, W):
        gu = torch.bmm(x, W)                       # [E, TPE, 2*INT]
        g, u = gu[..., :INTERMEDIATE], gu[..., INTERMEDIATE:]
        return F.silu(g) * u                       # [E, TPE, INT]  half-width out
    out["F4_unfused_bmm_then_swiglu"] = med_time(lambda: swiglu_from_gemm(xu, Wu), iters=20, warmup=8)
    cfn = torch.compile(swiglu_from_gemm)
    out["F4_compiled"] = med_time(lambda: cfn(xu, Wu), iters=20, warmup=8)
except Exception as ex:
    out["F4_error"] = str(ex)[:120]

with open("metax_glm.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: (round(v * 1e3, 4) if isinstance(v, float) else v) for k, v in out.items()}, indent=1))
print("[wrote metax_glm.json]")
