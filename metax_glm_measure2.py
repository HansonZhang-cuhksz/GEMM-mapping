"""Close the audit gap: measure F3 (rmsnorm+up_gate) and F5 (swiglu+down) fusions on C500, plus
in-place addmm for F1. F5 is the key memory-saver (avoids materializing activated[16384,2048])."""
import json, statistics
import torch
import torch.nn.functional as F

DEV = "cuda:0"
HIDDEN, INTERMEDIATE, EXPERTS, TPE = 6144, 2048, 256, 64


def med_time(fn, iters=30, warmup=12):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    return statistics.median(ts)


def bf(*s): return torch.randn(*s, device=DEV, dtype=torch.bfloat16)

out = {}

# ---- F3: rmsnorm(x) then up_gate bmm ----
xg, Wu, wn = bf(EXPERTS, TPE, HIDDEN), bf(EXPERTS, HIDDEN, 2 * INTERMEDIATE), bf(HIDDEN)
def rms(x): h = x.float(); return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16) * wn
def f3(x, W): return torch.bmm(rms(x), W)
out["F3_unfused"] = med_time(lambda: f3(xg, Wu), iters=20, warmup=8)
try:
    c3 = torch.compile(f3)
    out["F3_compiled"] = med_time(lambda: c3(xg, Wu), iters=20, warmup=8)
except Exception as ex: out["F3_err"] = str(ex)[:100]

# ---- F5: swiglu(gate,up) then down bmm (fusion avoids materializing activated) ----
gate, up, Wd = bf(EXPERTS, TPE, INTERMEDIATE), bf(EXPERTS, TPE, INTERMEDIATE), bf(EXPERTS, INTERMEDIATE, HIDDEN)
def f5(g, u, W): return torch.bmm(F.silu(g) * u, W)
out["F5_unfused"] = med_time(lambda: f5(gate, up, Wd), iters=20, warmup=8)  # eager: swiglu kernel + bmm
try:
    c5 = torch.compile(f5)
    out["F5_compiled"] = med_time(lambda: c5(gate, up, Wd), iters=20, warmup=8)
except Exception as ex: out["F5_err"] = str(ex)[:100]
# explicit unfused with materialized activated (what unfused really pays: write+read activated)
act = F.silu(gate) * up
out["F5_down_only"] = med_time(lambda: torch.bmm(act, Wd), iters=20, warmup=8)
out["F5_swiglu_only"] = med_time(lambda: F.silu(gate) * up, iters=40, warmup=12)

# ---- F1 in-place addmm (avoids the out-of-place D->D copy the audit found) ----
a_o, b_o, res = bf(2048, 16384), bf(16384, 6144), bf(2048, 6144)
out["F1_mm"] = med_time(lambda: a_o @ b_o, iters=20, warmup=8)
out["F1_addmm_outofplace"] = med_time(lambda: torch.addmm(res, a_o, b_o), iters=20, warmup=8)
buf = bf(2048, 6144)
out["F1_addmm_out_preallocated"] = med_time(lambda: torch.addmm(res, a_o, b_o, out=buf), iters=20, warmup=8)

with open("metax_glm2.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: (round(v * 1e3, 4) if isinstance(v, float) else v) for k, v in out.items()}, indent=1))
