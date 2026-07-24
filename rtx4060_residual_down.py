"""Round 5 (RTX4060_RESIDUAL_DOWN_TASK.md): residual2 fused into the dense DOWN-GEMM epilogue.

    out = residual2 + x_act @ W_down  =  addmm(residual2, x_act, W_down)     N = HIDDEN = 6144

The one untested residual site in the dense case. Tile-local epilogue -> estimator cell VALID;
stock fusion via cuBLASLt beta-accumulate expected (needs_custom_kernel = False predicted).

Two comparisons per row (the Round-4 lesson):
  (a) VENDOR (PRIMARY): best_unfused = min(eager mm+add, vendor mm + compiled-add clean variant);
      fused candidates = addmm (canonical), compiled/nocg/forced (core rows only), custom triton.
  (b) CUSTOM-vs-CUSTOM: same-family Triton GEMM +/- residual epilogue; best-vs-best and
      same-tile gains (mechanism isolation), exactly as rtx4060_n4_custom.py.

Sweep (host decision, asked): core-full + light rest -- all 6 paths on the 15 core configs
(K in {2048,6144,12288,16384,24576} x M in {2048,8192,32768}); the remaining 25 (M,K) rows run
the fast paths only (no whole-fn torch.compile; fields null with reason). M=131072 dropped
(x_act at K=24576 alone = 6.4 GB > 8 GB); covered by the per-K M-independence assertion over
M in {512, 8192, 32768}.

UNITS: med_time returns SECONDS; *_ms fields are ms. gain = unfused/fused (>1 => fusion faster).
Numerics: rel-vs-fp32 <= max(2*eager_rel, 5e-2), fp32 ref on a min(M,4096)-row slice
(slice-exact: GEMM rows independent, epilogue elementwise).

Run (from GEMM-mapping/):
    python rtx4060_residual_down.py --out rtx4060_residual_down.json --t2-json rtx4060_peak.json
    python rtx4060_residual_down.py --smoke --out /tmp/.../r5_smoke.json
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rtx4060_common import (  # noqa: E402
    ClockSampler,
    bf,
    geomean,
    med_time,
    save_json,
    sleep_cooldown,
)
from rtx4060_fusion_measure import (  # noqa: E402
    _cudagraph_copy_us,
    _rel_max,
    build_adjusted_profile,
    compile_and_time,
    judge_fused,
    profile_kernels,
    vendor_fused_ok,
)
import fusion_time_estimator as fte  # noqa: E402
from fusion_time_estimator import (  # noqa: E402
    Epilogue,
    _residual_aux,
    estimate_fused_gemm,
    estimate_gemm_grouped,
    estimate_vector_kernel,
)
from gemm_time_estimator import GPUS  # noqa: E402

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

HIDDEN, BPE = fte.HIDDEN, fte.BPE  # 6144, 2
ACT_MARKERS = ("add", "elementwise", "residual")


# --------------------------------------------------------------------------- #
# Estimator (spec section 3, verbatim; VALID cell -- tile-local epilogue)       #
# --------------------------------------------------------------------------- #
def est_res2_down_ms(M, K, gpu, N=HIDDEN):
    g = estimate_gemm_grouped("down", M, N, K, 1, gpu).time_s          # bare down GEMM
    res = estimate_vector_kernel("res2", 3 * M * N * BPE, gpu).time_s  # standalone add
    unf = g + res
    fus = estimate_fused_gemm("down+res2", M, N, K, 1,                 # residual read folded
                              Epilogue(extra_hbm_once=M * N * BPE,
                                       aux_smem_per_tile=_residual_aux), gpu).time_s
    return unf * 1e3, fus * 1e3, unf / fus


def est_sanity(gpu):
    """Spec section 3 anchors: ~1.159 @ (2048,2048), ~1.013 @ (2048,24576)."""
    _, _, g1 = est_res2_down_ms(2048, 2048, gpu)
    _, _, g2 = est_res2_down_ms(2048, 24576, gpu)
    ok = abs(g1 - 1.159) <= 0.01 and abs(g2 - 1.013) <= 0.01
    print(f"[est sanity] (2048,2048) gain={g1:.4f} (anchor 1.159) | "
          f"(2048,24576) gain={g2:.4f} (anchor 1.013) -> {'OK' if ok else 'DEVIATES'}",
          flush=True)
    return {"gain_2048_2048": g1, "gain_2048_24576": g2, "anchors_ok": ok}


# --------------------------------------------------------------------------- #
# Custom Triton kernel family (plain GEMM / residual-add / fused GEMM+residual) #
# --------------------------------------------------------------------------- #
@triton.jit
def _down_gemm_kernel(X, W, Y, M, N, K,
                      sxm, sxk, swk, swn, sym, syn,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N
    x_ptrs = X + offs_m[:, None] * sxm + offs_k[None, :] * sxk
    w_ptrs = W + offs_k[:, None] * swk + offs_n[None, :] * swn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kmask = offs_k < (K - k0)
        a = tl.load(x_ptrs, mask=m_mask & kmask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=kmask[:, None] & n_mask, other=0.0)
        acc += tl.dot(a, w)
        x_ptrs += BK * sxk
        w_ptrs += BK * swk
    tl.store(Y + offs_m[:, None] * sym + offs_n[None, :] * syn,
             acc.to(Y.dtype.element_ty), mask=m_mask & n_mask)


@triton.jit
def _res_add_kernel(Y, R, O, M, N, sym, syn, srm, srn, som, son,
                    BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y = tl.load(Y + offs_m[:, None] * sym + offs_n[None, :] * syn, mask=mask, other=0.0)
    r = tl.load(R + offs_m[:, None] * srm + offs_n[None, :] * srn, mask=mask, other=0.0)
    tl.store(O + offs_m[:, None] * som + offs_n[None, :] * son,
             (y.to(tl.float32) + r.to(tl.float32)).to(O.dtype.element_ty), mask=mask)


@triton.jit
def _down_gemm_res_kernel(X, W, R, O, M, N, K,
                          sxm, sxk, swk, swn, srm, srn, som, son,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # spec section 4: standard GEMM over K, residual2 folded into the store epilogue
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N
    x_ptrs = X + offs_m[:, None] * sxm + offs_k[None, :] * sxk
    w_ptrs = W + offs_k[:, None] * swk + offs_n[None, :] * swn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kmask = offs_k < (K - k0)
        a = tl.load(x_ptrs, mask=m_mask & kmask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=kmask[:, None] & n_mask, other=0.0)
        acc += tl.dot(a, w)
        x_ptrs += BK * sxk
        w_ptrs += BK * swk
    r = tl.load(R + offs_m[:, None] * srm + offs_n[None, :] * srn,
                mask=m_mask & n_mask, other=0.0)
    acc += r.to(tl.float32)
    tl.store(O + offs_m[:, None] * som + offs_n[None, :] * son,
             acc.to(O.dtype.element_ty), mask=m_mask & n_mask)


GEMM_CANDS = [(64, 64, 32, 4, 2), (128, 64, 32, 4, 3), (64, 128, 32, 4, 3),
              (128, 128, 32, 8, 3), (128, 128, 64, 8, 4), (64, 64, 64, 4, 3),
              (16, 64, 64, 4, 3), (32, 64, 32, 4, 2)]
ADD_CANDS = [(64, 128, 4), (32, 256, 4), (128, 64, 4)]


def c_gemm(x, W, y, cfg):
    BM, BN, BK, w, s = cfg
    M, K = x.shape
    N = W.shape[1]
    _down_gemm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        x, W, y, M, N, K, x.stride(0), x.stride(1), W.stride(0), W.stride(1),
        y.stride(0), y.stride(1), BM=BM, BN=BN, BK=BK, num_warps=w, num_stages=s)
    return y


def c_add(y, r, o, cfg):
    BM, BN, w = cfg
    M, N = y.shape
    _res_add_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        y, r, o, M, N, y.stride(0), y.stride(1), r.stride(0), r.stride(1),
        o.stride(0), o.stride(1), BM=BM, BN=BN, num_warps=w)
    return o


def c_fused(x, W, r, o, cfg):
    BM, BN, BK, w, s = cfg
    M, K = x.shape
    N = W.shape[1]
    _down_gemm_res_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        x, W, r, o, M, N, K, x.stride(0), x.stride(1), W.stride(0), W.stride(1),
        r.stride(0), r.stride(1), o.stride(0), o.stride(1),
        BM=BM, BN=BN, BK=BK, num_warps=w, num_stages=s)
    return o


def tune(make_call, cands):
    best = None
    for cfg in cands:
        try:
            t = med_time(make_call(cfg), iters=5, warmup=3)
        except Exception:
            continue
        if best is None or t < best[0]:
            best = (t, cfg)
    return best


# --------------------------------------------------------------------------- #
# Distinct code objects per whole-fn compile variant (core rows)                #
# --------------------------------------------------------------------------- #
def res_down_fn(x, W, r):
    return x @ W + r


def res_down_fn_nocg(x, W, r):
    return x @ W + r


def res_down_fn_forced(x, W, r):
    return x @ W + r


def make_add_only():
    def add_only(y, r):
        return y + r
    return add_only


# --------------------------------------------------------------------------- #
def measure_config(name, M, K, regime, core, iters, warmup, gpu, gpu_adj, smoke=False):
    N = HIDDEN if not smoke else 512
    print(f"[res2_down] {name}: M={M} N={N} K={K} ({regime}, {'CORE' if core else 'light'}) "
          f"...", flush=True)
    x, W, r = bf(M, K), bf(K, N), bf(M, N)
    y = torch.empty((M, N), device="cuda:0", dtype=torch.bfloat16)
    o = torch.empty((M, N), device="cuda:0", dtype=torch.bfloat16)

    ns = min(M, 4096)
    ref32 = x[:ns].float() @ W.float() + r[:ns].float()
    eager = x @ W + r
    eager_rel = _rel_max(eager[:ns], ref32)
    rel_tol = max(2.0 * eager_rel, 5e-2)
    del eager

    row = {"name": name, "kind": "res2_down", "regime": regime, "core": core,
           "dims": {"M": M, "N": N, "K": K}}
    with ClockSampler() as cs:
        vendor_gemm_s = med_time(lambda: x @ W, iters, warmup)
        eager_unf_s = med_time(lambda: x @ W + r, iters, warmup)

        # clean unfused variant: vendor mm + COMPILED add (cheap pointwise compile, all rows)
        add_res = compile_and_time(make_add_only(), (x @ W, r), iters, warmup,
                                   "max-autotune-no-cudagraphs")
        clean_unf_s = None
        if add_res["ms"] is not None:
            cadd = torch.compile(make_add_only(), mode="max-autotune-no-cudagraphs",
                                 dynamic=False)
            cadd(x @ W, r)
            clean_unf_s = med_time(lambda: cadd(x @ W, r), iters, warmup)

        # addmm: the canonical stock fused path (all rows) + kernel evidence
        addmm_s = med_time(lambda: torch.addmm(r, x, W), iters, warmup)
        addmm_out = torch.addmm(r, x, W)
        addmm_rel = _rel_max(addmm_out[:ns], ref32)
        addmm_ok = bool(addmm_rel <= rel_tol)
        addmm_kernels = profile_kernels(lambda: torch.addmm(r, x, W))
        addmm_fused, addmm_sep = vendor_fused_ok(addmm_kernels, ACT_MARKERS)
        del addmm_out

        # whole-fn compile variants: CORE rows only (the expensive GEMM autotunes)
        if core:
            big = M * K * 2 * 2 + M * N * 2 * 3 + K * N * 2 * 2 > 7 * 2**30
            r_def = (compile_and_time(res_down_fn, (x, W, r), iters, warmup, "max-autotune")
                     if not big else {"ms": None, "kernels": [], "fused": False,
                                      "evidence": {}, "numerics_ok": None, "max_abs": None,
                                      "error": "skipped: cudagraph static copies would "
                                               "exceed ~7 GiB at this footprint"})
            r_nocg = compile_and_time(res_down_fn_nocg, (x, W, r), iters, warmup,
                                      "max-autotune-no-cudagraphs")
            r_forced = compile_and_time(res_down_fn_forced, (x, W, r), iters, warmup,
                                        "max-autotune-no-cudagraphs", forced=True)
            for rr in (r_def, r_nocg, r_forced):
                out_c = rr.pop("_cfn_out", None)
                if out_c is not None:
                    rr["numerics_ok"] = bool(_rel_max(out_c[:ns], ref32) <= rel_tol)
                if rr["kernels"]:
                    rr["fused"], rr["evidence"] = judge_fused(rr["kernels"], ACT_MARKERS)
                    vok, vsep = vendor_fused_ok(rr["kernels"], ACT_MARKERS)
                    rr["evidence"]["vendor_fused_ok"] = vok
                    rr["evidence"]["vendor_separate_kernels"] = vsep
        else:
            skip = {"ms": None, "kernels": [], "fused": False, "evidence": {},
                    "numerics_ok": None, "max_abs": None,
                    "error": "not run: light row (host-approved core-full + light-rest sweep)"}
            r_def, r_nocg, r_forced = dict(skip), dict(skip), dict(skip)

        # custom-vs-custom (all rows)
        tg = tune(lambda c: (lambda: c_gemm(x, W, y, c)), GEMM_CANDS if not smoke else GEMM_CANDS[:3])
        ta = tune(lambda c: (lambda: c_add(y, r, o, c)), ADD_CANDS if not smoke else ADD_CANDS[:2])
        tf = tune(lambda c: (lambda: c_fused(x, W, r, o, c)), GEMM_CANDS if not smoke else GEMM_CANDS[:3])
        if tg is None or ta is None or tf is None:
            raise RuntimeError(f"custom kernels failed to tune (gemm={tg} add={ta} fused={tf})")
        g_cfg, a_cfg, f_cfg = tg[1], ta[1], tf[1]
        cust_unf_s = med_time(lambda: (c_gemm(x, W, y, g_cfg), c_add(y, r, o, a_cfg)),
                              iters, warmup)
        cust_fus_s = med_time(lambda: c_fused(x, W, r, o, f_cfg), iters, warmup)
        c_fused(x, W, r, o, f_cfg)
        cust_rel = _rel_max(o[:ns], ref32)
        cust_ok = bool(cust_rel <= rel_tol)
        c_gemm(x, W, y, g_cfg)
        c_add(y, r, o, a_cfg)
        cunf_rel = _rel_max(o[:ns], ref32)
        cunf_ok = bool(cunf_rel <= rel_tol)
        try:
            cust_unf_st_s = med_time(lambda: (c_gemm(x, W, y, f_cfg), c_add(y, r, o, a_cfg)),
                                     iters, warmup)
        except Exception:
            cust_unf_st_s = None

        gemm_repeat_s = med_time(lambda: x @ W, iters, warmup)
    clocks = cs.summary()
    sleep_cooldown()

    # ---- assemble ----
    eager_unf_ms = eager_unf_s * 1e3
    clean_unf_ms = clean_unf_s and clean_unf_s * 1e3
    best_unfused_ms = min(v for v in (eager_unf_ms, clean_unf_ms) if v is not None)
    addmm_ms = addmm_s * 1e3
    fused_paths = {"addmm_ms": addmm_ms, "compiled_ms": r_def["ms"],
                   "nocg_ms": r_nocg["ms"], "forced_ms": r_forced["ms"],
                   "triton_ms": cust_fus_s * 1e3}
    cand = [v for v in fused_paths.values() if v is not None]
    best_fused_ms = min(cand)
    measured_gain = best_unfused_ms / best_fused_ms
    vcand = []
    if addmm_ok and addmm_fused:
        vcand.append(addmm_ms)
    for rr in (r_def, r_nocg, r_forced):
        if rr["ms"] is not None and rr.get("numerics_ok") and (
                rr.get("fused") or rr.get("evidence", {}).get("vendor_fused_ok")):
            vcand.append(rr["ms"])
    stock_fused_verified = bool(vcand)
    if cust_ok:
        vcand.append(cust_fus_s * 1e3)
    measured_gain_verified = (best_unfused_ms / min(vcand)) if vcand else None

    drift = gemm_repeat_s / vendor_gemm_s
    est_unf, est_fus, est_gain = est_res2_down_ms(M, K, gpu, N=N)
    row.update({
        "vendor_gemm_only_ms": vendor_gemm_s * 1e3,
        "vendor_eager_unfused_ms": eager_unf_ms,
        "clean_unfused_ms": clean_unf_ms,
        "best_unfused_ms": best_unfused_ms,
        "addmm_ms": addmm_ms, "compiled_ms": r_def["ms"], "nocg_ms": r_nocg["ms"],
        "forced_ms": r_forced["ms"], "triton_ms": cust_fus_s * 1e3,
        "best_fused_ms": best_fused_ms,
        "addmm_fused_verified": addmm_fused,
        "addmm_separate_kernels": addmm_sep,
        "addmm_kernel_evidence": addmm_kernels,
        "fused_verified": addmm_fused or any(
            rr.get("fused") or rr.get("evidence", {}).get("vendor_fused_ok")
            for rr in (r_def, r_nocg, r_forced) if rr["ms"] is not None),
        "kernel_evidence": {"compiled": r_def["kernels"], "nocg": r_nocg["kernels"],
                            "forced": r_forced["kernels"]},
        "fusion_evidence": {"compiled": r_def["evidence"], "nocg": r_nocg["evidence"],
                            "forced": r_forced["evidence"]},
        "compile_errors": {"compiled": r_def["error"], "nocg": r_nocg["error"],
                           "forced": r_forced["error"], "add_only": add_res["error"]},
        "compiled_cudagraph_input_copy_us": _cudagraph_copy_us(r_def["kernels"]),
        "stock_fused_verified": stock_fused_verified,
        "needs_custom_kernel": not stock_fused_verified,
        "measured_gain": measured_gain,
        "measured_gain_verified": measured_gain_verified,
        "custom_unfused_ms": cust_unf_s * 1e3,
        "custom_unfused_cfg": {"gemm": g_cfg, "add": a_cfg},
        "custom_fused_ms": cust_fus_s * 1e3, "custom_fused_cfg": f_cfg,
        "custom_gain_best": cust_unf_s / cust_fus_s,
        "custom_unfused_same_tile_ms": cust_unf_st_s and cust_unf_st_s * 1e3,
        "custom_gain_same_tile": cust_unf_st_s and cust_unf_st_s / cust_fus_s,
        "custom_gemm_over_vendor_gemm": tg[0] / vendor_gemm_s,
        "est_unfused_ms": est_unf, "est_fused_ms": est_fus, "estimated_gain": est_gain,
        "numerics": {"eager_rel_max_vs_fp32": eager_rel, "rel_tol": rel_tol,
                     "rows_checked": ns, "addmm_rel": addmm_rel, "addmm_ok": addmm_ok,
                     "custom_fused_rel": cust_rel, "custom_fused_ok": cust_ok,
                     "custom_unfused_rel": cunf_rel, "custom_unfused_ok": cunf_ok,
                     "compiled_ok": r_def.get("numerics_ok"),
                     "nocg_ok": r_nocg.get("numerics_ok"),
                     "forced_ok": r_forced.get("numerics_ok")},
        "gemm_drift_ratio": drift,
        "drift_clean": bool(abs(drift - 1) <= 0.05),
        "excluded_from_aggregate": bool(abs(drift - 1) > 0.05),
        "clocks": clocks,
    })
    if gpu_adj is not None:
        a_unf, a_fus, a_gain = est_res2_down_ms(M, K, gpu_adj, N=N)
        row["est_unfused_ms_adj"], row["est_fused_ms_adj"] = a_unf, a_fus
        row["estimated_gain_adj"] = a_gain
    del x, W, r, y, o, ref32
    torch.cuda.empty_cache()
    print(f"    unf={best_unfused_ms:.4f} addmm={addmm_ms:.4f} "
          f"(fused={addmm_fused}) gain={measured_gain:.4f} "
          f"gain_ver={measured_gain_verified and round(measured_gain_verified, 4)} "
          f"est={est_gain:.4f} custom_best={row['custom_gain_best']:.4f} "
          f"same_tile={row['custom_gain_same_tile'] and round(row['custom_gain_same_tile'], 4)} "
          f"needs_custom={row['needs_custom_kernel']} drift={drift:.3f}", flush=True)
    return row


KS = (2048, 6144, 12288, 16384, 24576)
MS = (512, 1024, 2048, 4096, 8192, 16384, 32768, 49152)
CORE_MS = (2048, 8192, 32768)
MI_MS = (512, 8192, 32768)  # per-K M-independence assertion points


def configs(smoke):
    if smoke:
        return [("r5_smoke_core", 256, 512, "decode", True),
                ("r5_smoke_light", 512, 512, "decode", False)]
    out = []
    for K in KS:
        for M in MS:
            regime = "decode" if M <= 16384 else "prefill"
            out.append((f"r5_M{M}_K{K}", M, K, regime, M in CORE_MS))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--t2-json", default=None)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    iters, warmup = (3, 3) if args.smoke else (30, 15)
    gpu = GPUS["rtx4060-measured"]
    gpu_adj, adj_info = (None, None)
    if args.t2_json:
        gpu_adj, adj_info = build_adjusted_profile(args.t2_json, gpu)

    env = {
        "task": "Round 5: residual2 -> dense down-GEMM epilogue (RTX4060_RESIDUAL_DOWN_TASK.md)",
        "torch": torch.__version__, "triton": triton.__version__,
        "device": torch.cuda.get_device_name(0),
        "clocks_locked": True,
        "clocks_note": "clocks LOCKED 1500/5501 by host; ClockSampler per config",
        "sweep": "core-full + light rest (host decision): all 6 paths on 5K x M in {2048,8192,"
                 "32768}; light rows skip the whole-fn compile variants (null with reason)",
        "M_131072_dropped": "x_act[131072,24576] bf16 = 6.4 GB alone > 8 GB; covered by the "
                            "per-K M-independence assertion (M in {512,8192,32768})",
        "estimator_cell": "VALID (tile-local epilogue) - NOT structure-blind",
        "est_sanity": est_sanity(gpu),
        "smoke": args.smoke, "adjusted_profile": adj_info,
    }
    conventions = {
        "units": "ms for *_ms; gain = unfused/fused (>1 => fusion faster)",
        "best_unfused_ms": "min(eager mm+add, vendor mm + compiled-add clean variant)",
        "measured_gain_verified": "best_unfused / best VERIFIED-fused (addmm iff profiler shows "
                                  "no separate add kernel + numerics; compiled/nocg/forced iff "
                                  "template- or vendor-fused per evidence + numerics; custom "
                                  "triton fused iff numerics)",
        "needs_custom_kernel": "False iff a STOCK path (addmm/compiled/nocg/forced) is "
                               "verified-fused",
        "custom_gain_*": "custom-vs-custom within one Triton tiling family (Round-4 lesson): "
                         "best-vs-best and same-tile (unfused GEMM forced onto the fused "
                         "kernel's tile)",
        "timing": f"median of {iters} cuda events after {warmup} warmup + clock warmer; "
                  "drift probe = bare vendor mm re-measured at config END",
        "numerics": "rel-vs-fp32 <= max(2*eager_rel, 5e-2) on a min(M,4096)-row slice",
    }
    results = []
    out = {"conventions": conventions, "env": env, "configs": results}
    for name, M, K, regime, core in configs(args.smoke):
        try:
            row = measure_config(name, M, K, regime, core, iters, warmup, gpu, gpu_adj,
                                 smoke=args.smoke)
        except Exception as exc:
            traceback.print_exc()
            row = {"name": name, "kind": "res2_down", "regime": regime, "core": core,
                   "error": f"{type(exc).__name__}: {exc}"}
        results.append(row)
        save_json(args.out, out)

    # ---- aggregate ----
    ok = [r for r in results if "error" not in r]
    agg = {"per_K": {}, "gain_token_independent_perK": {}}
    for K in (KS if not args.smoke else [512]):
        sel = [r for r in ok if r["dims"]["K"] == K]
        cl = [r for r in sel if r["drift_clean"]]
        rl = [r for r in sel if abs(r["gemm_drift_ratio"] - 1) <= 0.10]
        e = geomean([r["estimated_gain"] for r in sel]) if sel else None
        entry = {"n": len(sel), "n_clean": len(cl), "n_relaxed": len(rl),
                 "est_gm_allM": e}
        for label, ss in (("clean", cl), ("relaxed", rl)):
            if ss:
                gv = geomean([r["measured_gain_verified"] for r in ss
                              if r["measured_gain_verified"]])
                eg = geomean([r["estimated_gain"] for r in ss])
                entry[label] = {"gain_ver_gm": gv, "est_gm": eg,
                                "delivered": (gv - 1) / (eg - 1) if gv and eg and eg != 1 else None,
                                "custom_best_gm": geomean([r["custom_gain_best"] for r in ss]),
                                "custom_same_tile_gm": geomean(
                                    [r["custom_gain_same_tile"] for r in ss
                                     if r["custom_gain_same_tile"]])}
        agg["per_K"][str(K)] = entry
        mi = {r["dims"]["M"]: r["measured_gain_verified"] for r in sel
              if r["dims"]["M"] in MI_MS and r["measured_gain_verified"]}
        if len(mi) >= 2:
            vals = list(mi.values())
            spread = (max(vals) - min(vals)) / min(vals)
            agg["gain_token_independent_perK"][str(K)] = {
                "gains_by_M": {str(m): g for m, g in mi.items()},
                "spread_frac": spread, "flat": bool(spread <= 0.05),
                "n_points": len(mi)}
    for label in ("clean", "relaxed"):
        sel = [r for r in ok if (r["drift_clean"] if label == "clean"
                                 else abs(r["gemm_drift_ratio"] - 1) <= 0.10)]
        if sel:
            gv = geomean([r["measured_gain_verified"] for r in sel
                          if r["measured_gain_verified"]])
            eg = geomean([r["estimated_gain"] for r in sel])
            agg[f"overall_{label}"] = {
                "n": len(sel), "gain_ver_gm": gv, "est_gm": eg,
                "delivered": (gv - 1) / (eg - 1) if gv and eg != 1 else None,
                "custom_best_gm": geomean([r["custom_gain_best"] for r in sel]),
                "custom_same_tile_gm": geomean([r["custom_gain_same_tile"] for r in sel
                                                if r["custom_gain_same_tile"]])}
    agg["stock_fused_verified_count"] = sum(1 for r in ok if r["stock_fused_verified"])
    agg["needs_custom_kernel_any"] = any(r["needs_custom_kernel"] for r in ok)
    out["aggregate"] = agg
    save_json(args.out, out)

    print("\n=== R5 residual2->down summary ===")
    print(f"{'config':<20}{'K':>7}{'M':>7}{'gain_ver':>9}{'est':>7}{'deliv':>7}"
          f"{'cust_b':>8}{'same_t':>8}{'addmm?':>7}{'drift':>7}")
    for r in ok:
        gv = r["measured_gain_verified"]
        dl = (gv - 1) / (r["estimated_gain"] - 1) if gv and r["estimated_gain"] != 1 else None
        print(f"{r['name']:<20}{r['dims']['K']:>7}{r['dims']['M']:>7}"
              f"{(gv or float('nan')):>9.4f}{r['estimated_gain']:>7.3f}"
              f"{(dl if dl is not None else float('nan')):>7.2f}"
              f"{r['custom_gain_best']:>8.4f}"
              f"{(r['custom_gain_same_tile'] or float('nan')):>8.4f}"
              f"{str(r['addmm_fused_verified'])[:1]:>7}{r['gemm_drift_ratio']:>7.3f}")
    print("aggregate:", {k: v for k, v in agg.items() if k.startswith('overall')})


if __name__ == "__main__":
    main()
