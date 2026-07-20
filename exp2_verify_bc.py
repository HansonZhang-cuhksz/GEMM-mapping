"""EXPERIMENT 2 parts (a-inplace), (b), (c) on physical MetaX C500 (fusion env, bf16, cuda:0)."""
import json, statistics
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

DEV = "cuda:0"
HIDDEN, INTERMEDIATE, EXPERTS = 6144, 2048, 256
BATCH, TOP_K = 2048, 8
TPE = BATCH * TOP_K // EXPERTS   # 64
KV = 64 * 256                    # 16384
PEAK_BW = 1.43e12                # measured C500 HBM peak B/s
GiB = 2**30


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
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    kevs = [e for e in prof.events() if getattr(e, "device_time", 0) and e.device_time > 0]
    agg = {}
    for e in kevs:
        agg.setdefault(e.key, [0, 0.0])
        agg[e.key][0] += e.count if e.count else 1
        agg[e.key][1] += e.device_time
    per_call = sum(v[0] for v in agg.values()) / reps
    names = [(k[:55], round(v[1] / reps, 2)) for k, v in sorted(agg.items(), key=lambda kv: -kv[1][1])]
    return per_call, names


out = {}

# =====================================================================================
# (a-inplace) does in-place addmm_ skip the D2D staging copy? -> pure beta-accum epilogue cost
# =====================================================================================
print("=" * 78)
print("(a2) in-place accumulate: res.addmm_(a,b) vs res += a@b  (mla_o shape)")
print("=" * 78)
a_o, b_o = bf(BATCH, KV), bf(KV, HIDDEN)


def run_addmm_inplace():
    r = bf(BATCH, HIDDEN)             # fresh accumulator each call (fair vs out-of-place)
    r.addmm_(a_o, b_o); return r


def run_iadd():
    r = bf(BATCH, HIDDEN)
    r += a_o @ b_o; return r


k_ip, n_ip = count_kernels(run_addmm_inplace)
k_ia, n_ia = count_kernels(run_iadd)
print(f"kernels/call:  addmm_(in-place)={k_ip:.1f}   (a@b then +=)={k_ia:.1f}")
print("  addmm_ kernels:", n_ip)
print("  +=     kernels:", n_ia)
# time only the accumulate op (exclude the fresh-alloc/randn) by pre-allocating and resetting cheaply
r_acc = bf(BATCH, HIDDEN)
t_addmm_ip = med_time(lambda: r_acc.addmm_(a_o, b_o), iters=30, warmup=10)
r_acc2 = bf(BATCH, HIDDEN)
t_iadd = med_time(lambda: r_acc2.add_(a_o @ b_o), iters=30, warmup=10)
print(f"times(ms): addmm_(fused,in-place)={t_addmm_ip*1e3:.4f}   a@b+add_={t_iadd*1e3:.4f}"
      f"   -> in-place addmm vs add {t_iadd/t_addmm_ip:.4f}x")
out["a2_kernels_addmm_inplace"] = k_ip
out["a2_t_addmm_inplace"] = t_addmm_ip
out["a2_t_iadd"] = t_iadd

# =====================================================================================
# (b) torch.compile: does the metax backend fuse anything?
# =====================================================================================
print("\n" + "=" * 78)
print("(b) torch.compile on MetaX backend")
print("=" * 78)

# (b1) sanity: KNOWN-fusable elementwise chain (x.sin()+1).cos()*2
x = bf(BATCH * TOP_K, INTERMEDIATE)   # 16384 x 2048


def elt_chain(t):
    return (t.sin() + 1).cos() * 2


try:
    ce = torch.compile(elt_chain)
    y_e = elt_chain(x); y_c = ce(x)
    ediff = (y_e.float() - y_c.float()).abs().max().item()
    k_elt_eager, n_ee = count_kernels(lambda: elt_chain(x))
    k_elt_comp, n_ec = count_kernels(lambda: ce(x))
    t_ee = med_time(lambda: elt_chain(x), iters=40, warmup=15)
    t_ec = med_time(lambda: ce(x), iters=40, warmup=15)
    print(f"elementwise (x.sin()+1).cos()*2  numeric max_diff={ediff:.3g}")
    print(f"  kernels/call: eager={k_elt_eager:.1f}  compiled={k_elt_comp:.1f}   "
          f"({'FUSED' if k_elt_comp < k_elt_eager else 'NOT fused'})")
    print("  eager   :", n_ee)
    print("  compiled:", n_ec)
    print(f"  time(ms): eager={t_ee*1e3:.4f}  compiled={t_ec*1e3:.4f}  speedup {t_ee/t_ec:.3f}x")
    out["b_elt_kernels_eager"] = k_elt_eager
    out["b_elt_kernels_compiled"] = k_elt_comp
    out["b_elt_t_eager"] = t_ee; out["b_elt_t_compiled"] = t_ec
except Exception as ex:
    print("  elementwise compile FAILED:", repr(ex)[:200])
    out["b_elt_error"] = repr(ex)[:200]

# (b2) graph-break check via dynamo.explain on the F4 function
xu = bf(EXPERTS, TPE, HIDDEN)
Wu = bf(EXPERTS, HIDDEN, 2 * INTERMEDIATE)


def f4(x, W):
    gu = torch.bmm(x, W)
    g, u = gu[..., :INTERMEDIATE], gu[..., INTERMEDIATE:]
    return F.silu(g) * u


try:
    import torch._dynamo as dynamo
    exp = dynamo.explain(f4)(xu, Wu)
    print(f"\nF4 dynamo.explain: graph_count={exp.graph_count} graph_break_count={exp.graph_break_count} "
          f"op_count={exp.op_count}")
    out["b_f4_graph_count"] = exp.graph_count
    out["b_f4_graph_break_count"] = exp.graph_break_count
    if exp.break_reasons:
        for br in exp.break_reasons[:4]:
            print("   break:", str(getattr(br, 'reason', br))[:90])
except Exception as ex:
    print("  dynamo.explain FAILED:", repr(ex)[:200])
    out["b_f4_explain_error"] = repr(ex)[:200]

# (b3) F4: eager vs compiled kernel counts + hand-fused epilogue traffic
try:
    cf4 = torch.compile(f4)
    y_f4e = f4(xu, Wu); y_f4c = cf4(xu, Wu)
    f4diff = (y_f4e.float() - y_f4c.float()).abs().max().item()
    k_f4e, n_f4e = count_kernels(lambda: f4(xu, Wu))
    k_f4c, n_f4c = count_kernels(lambda: cf4(xu, Wu))
    t_f4e = med_time(lambda: f4(xu, Wu), iters=20, warmup=8)
    t_f4c = med_time(lambda: cf4(xu, Wu), iters=20, warmup=8)
    print(f"\nF4 bmm+swiglu  numeric max_diff={f4diff:.3g}")
    print(f"  kernels/call: eager={k_f4e:.1f}  compiled={k_f4c:.1f}   "
          f"({'compiler fused GEMM+epilogue' if k_f4c < k_f4e else 'NO GEMM+epilogue fusion'})")
    print("  eager   :", n_f4e)
    print("  compiled:", n_f4c)
    print(f"  time(ms): eager={t_f4e*1e3:.4f}  compiled={t_f4c*1e3:.4f}")
    out["b_f4_kernels_eager"] = k_f4e; out["b_f4_kernels_compiled"] = k_f4c
    out["b_f4_t_eager"] = t_f4e; out["b_f4_t_compiled"] = t_f4c
except Exception as ex:
    print("  F4 compile FAILED:", repr(ex)[:200])
    out["b_f4_error"] = repr(ex)[:200]

# (b4) hand-fused swiglu epilogue traffic accounting
gu = bf(EXPERTS, TPE, 2 * INTERMEDIATE)   # the intermediate GEMM output


def swiglu_epi(gu):
    g, u = gu[..., :INTERMEDIATE], gu[..., INTERMEDIATE:]
    return F.silu(g) * u


k_epi, n_epi = count_kernels(lambda: swiglu_epi(gu))
t_epi = med_time(lambda: swiglu_epi(gu), iters=40, warmup=15)
gu_bytes = EXPERTS * TPE * 2 * INTERMEDIATE * 2
out_bytes = EXPERTS * TPE * INTERMEDIATE * 2
print(f"\nhand-fused swiglu epilogue (silu(g)*u, half-width out):")
print(f"  kernels/call={k_epi:.1f}  time(ms)={t_epi*1e3:.4f}   {n_epi}")
print(f"  gu(read) {gu_bytes/GiB:.3f} GiB | out(write) {out_bytes/GiB:.3f} GiB")
print(f"  UNFUSED extra HBM traffic (write gu + read gu back) = {2*gu_bytes/GiB:.3f} GiB "
      f"= {2*gu_bytes/PEAK_BW*1e3:.4f} ms max saveable by GEMM+epilogue fusion")
out["b_swiglu_epi_kernels"] = k_epi; out["b_swiglu_epi_t"] = t_epi
out["b_gu_gib"] = gu_bytes / GiB
out["b_fusion_saveable_ms"] = 2 * gu_bytes / PEAK_BW * 1e3

# =====================================================================================
# (c) grouped FFN weight-bandwidth-bound?
# =====================================================================================
print("\n" + "=" * 78)
print("(c) grouped FFN effective bandwidth vs 1.43 TB/s peak")
print("=" * 78)
T_UP, T_DN = 0.00930790376663208, 0.004628096103668213   # from metax_glm.json
w_up = EXPERTS * HIDDEN * 2 * INTERMEDIATE * 2
w_dn = EXPERTS * INTERMEDIATE * HIDDEN * 2
# activation traffic
a_up = EXPERTS * TPE * HIDDEN * 2 + EXPERTS * TPE * 2 * INTERMEDIATE * 2   # in + out
a_dn = EXPERTS * TPE * INTERMEDIATE * 2 + EXPERTS * TPE * HIDDEN * 2
for tag, w, act, t in [("up_gate", w_up, a_up, T_UP), ("down", w_dn, a_dn, T_DN)]:
    bw_w = w / t
    bw_tot = (w + act) / t
    print(f"  {tag}: W={w/GiB:.3f} GiB, act={act/GiB:.3f} GiB, t={t*1e3:.4f} ms")
    print(f"     weight-only BW = {bw_w/1e12:.3f} TB/s ({100*bw_w/PEAK_BW:.1f}% of peak) | "
          f"total(W+act) BW = {bw_tot/1e12:.3f} TB/s ({100*bw_tot/PEAK_BW:.1f}% of peak)")
    out[f"c_{tag}_w_gib"] = w / GiB
    out[f"c_{tag}_bw_weight_tbs"] = bw_w / 1e12
    out[f"c_{tag}_bw_weight_pct"] = 100 * bw_w / PEAK_BW
    out[f"c_{tag}_bw_total_pct"] = 100 * bw_tot / PEAK_BW

# layer breakdown & vector-fusion ceiling
comp = {"mla_o": 1.935, "router": 0.070, "up_gate": 9.308, "down": 4.628,
        "residual": 0.063, "rmsnorm": 0.373, "swiglu": 0.256}
layer = sum(comp.values())
ffn = comp["up_gate"] + comp["down"] + comp["swiglu"]
vec = comp["residual"] + comp["rmsnorm"] + comp["swiglu"]
print(f"\n  layer total (sum of kernels) = {layer:.3f} ms")
print(f"  FFN (up_gate+down+swiglu) = {ffn:.3f} ms = {100*ffn/layer:.1f}% of layer")
print(f"  up_gate+down (weight-BW-bound) = {comp['up_gate']+comp['down']:.3f} ms = "
      f"{100*(comp['up_gate']+comp['down'])/layer:.1f}% of layer")
print(f"  ALL vector kernels (resid+rms+swiglu) = {vec:.3f} ms = {100*vec/layer:.2f}% of layer")
print(f"  => fusing every vector kernel away saves at most {100*vec/layer:.2f}% of the layer")
out["c_layer_ms"] = layer
out["c_ffn_pct"] = 100 * ffn / layer
out["c_vector_pct_ceiling"] = 100 * vec / layer

with open("exp2_verify_bc.json", "w") as f:
    json.dump(out, f, indent=1)
print("\n[wrote exp2_verify_bc.json]")
