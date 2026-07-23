"""N4 addendum (host request, round 4): custom-FUSED vs custom-UNFUSED only.

The T4/T6.D N4 (SwiGLU->up_gate, F4) comparisons judged the fused hand Triton kernel against
VENDOR baselines (cuBLAS GEMM + eager/compiled epilogue), conflating (fusion benefit) with
(Triton-vs-cuBLAS GEMM-quality gap). Here BOTH sides come from the same custom Triton tiling
family, isolating the fusion effect:

  unfused_custom = triton_gemm_full (writes the full [M, 2*inter] gu)
                 + triton_swiglu_elem (separate elementwise: silu(g)*u, reads gu, writes act)
  fused_custom   = triton_swiglu_fused (dual-accumulator, writes act directly; the same
                   kernel design measured in T4/T6.D)

All three kernels are batched-capable (expert dim via stride; dense = E=1) so the MoE/grouped
case gets a REAL custom fused path for the first time (T4's --moe rows had none).

Reported per config:
  * best-vs-best:  min-over-tiles(unfused pair) / min-over-tiles(fused)   [realizable gain]
  * same-tile:     unfused and fused forced to the SAME (BM,BN,BK,w,s)    [mechanism isolation]
  * vendor context (cuBLAS gemm_only + eager unfused) -- context columns ONLY, not the verdict
  * est gain via est_swiglu_ms -- the estimator assumes the same optimal-mapping GEMM on both
    sides, i.e. custom-vs-custom is exactly the estimator's regime: this is the cleanest
    est-vs-measured fusion test in the study.

UNITS: med_time returns SECONDS; every *_ms field is ms. gain = unfused/fused (>1 => fusion
faster). Numerics: rel-vs-fp32 <= max(2*eager_rel, 5e-2) per T6.0.4.

Run (from GEMM-mapping/):
    python rtx4060_n4_custom.py --out rtx4060_n4_custom.json --t2-json rtx4060_peak.json
    python rtx4060_n4_custom.py --smoke --out /tmp/.../n4_smoke.json
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

import torch
import torch.nn.functional as F

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
    _rel_max,
    build_adjusted_profile,
    est_swiglu_ms,
)
from gemm_time_estimator import GPUS  # noqa: E402

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


# --------------------------------------------------------------------------- #
# The custom kernel family (shared tiling; batched via expert stride)           #
# --------------------------------------------------------------------------- #
@triton.jit
def _gemm_full_kernel(X, W, GU, M, N, K,
                      sxe, sxm, sxk, swe, swk, swn, sge, sgm, sgn,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # plain GEMM writing the FULL [M, N=2*inter] gu (the custom UNFUSED stage 1)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.program_id(2)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N
    x_ptrs = X + e * sxe + offs_m[:, None] * sxm + offs_k[None, :] * sxk
    w_ptrs = W + e * swe + offs_k[:, None] * swk + offs_n[None, :] * swn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kmask = offs_k < (K - k0)
        a = tl.load(x_ptrs, mask=m_mask & kmask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=kmask[:, None] & n_mask, other=0.0)
        acc += tl.dot(a, w)
        x_ptrs += BK * sxk
        w_ptrs += BK * swk
    tl.store(GU + e * sge + offs_m[:, None] * sgm + offs_n[None, :] * sgn,
             acc.to(GU.dtype.element_ty), mask=m_mask & n_mask)


@triton.jit
def _swiglu_elem_kernel(GU, O, M, INTER,
                        sge, sgm, sgn, soe, som, son,
                        BM: tl.constexpr, BN: tl.constexpr):
    # separate custom elementwise: act = silu(gu[:, :inter]) * gu[:, inter:]  (stage 2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.program_id(2)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < INTER
    g = tl.load(GU + e * sge + offs_m[:, None] * sgm + offs_n[None, :] * sgn,
                mask=m_mask & n_mask, other=0.0).to(tl.float32)
    u = tl.load(GU + e * sge + offs_m[:, None] * sgm + (offs_n[None, :] + INTER) * sgn,
                mask=m_mask & n_mask, other=0.0).to(tl.float32)
    y = (g * tl.sigmoid(g)) * u
    tl.store(O + e * soe + offs_m[:, None] * som + offs_n[None, :] * son,
             y.to(O.dtype.element_ty), mask=m_mask & n_mask)


@triton.jit
def _swiglu_fused_kernel(X, W, O, M, INTER, K,
                         sxe, sxm, sxk, swe, swk, swn, soe, som, son,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # dual-accumulator fused GEMM+SwiGLU (same design as T4's hand kernel; batched)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.program_id(2)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < INTER
    x_ptrs = X + e * sxe + offs_m[:, None] * sxm + offs_k[None, :] * sxk
    wg_ptrs = W + e * swe + offs_k[:, None] * swk + offs_n[None, :] * swn
    wu_ptrs = W + e * swe + offs_k[:, None] * swk + (offs_n[None, :] + INTER) * swn
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kmask = offs_k < (K - k0)
        a = tl.load(x_ptrs, mask=m_mask & kmask[None, :], other=0.0)
        wg = tl.load(wg_ptrs, mask=kmask[:, None] & n_mask, other=0.0)
        wu = tl.load(wu_ptrs, mask=kmask[:, None] & n_mask, other=0.0)
        acc_g += tl.dot(a, wg)
        acc_u += tl.dot(a, wu)
        x_ptrs += BK * sxk
        wg_ptrs += BK * swk
        wu_ptrs += BK * swk
    y = (acc_g * tl.sigmoid(acc_g)) * acc_u
    tl.store(O + e * soe + offs_m[:, None] * som + offs_n[None, :] * son,
             y.to(O.dtype.element_ty), mask=m_mask & n_mask)


GEMM_CANDS = [(64, 64, 32, 4, 2), (128, 64, 32, 4, 3), (64, 128, 32, 4, 3),
              (128, 128, 32, 8, 3), (128, 128, 64, 8, 4), (64, 64, 64, 4, 3),
              (16, 64, 64, 4, 3), (32, 64, 32, 4, 2)]     # small-BM entries for tpe<=32
ELEM_CANDS = [(64, 128, 4), (32, 256, 4), (128, 64, 4), (64, 64, 2)]


def _strides3(t):
    return (t.stride(0), t.stride(1), t.stride(2)) if t.dim() == 3 else (0, t.stride(0), t.stride(1))


def gemm_full(x, W, gu, cfg):
    BM, BN, BK, w, s = cfg
    E = x.shape[0] if x.dim() == 3 else 1
    M, K = x.shape[-2], x.shape[-1]
    N = W.shape[-1]
    _gemm_full_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN), E)](
        x, W, gu, M, N, K, *_strides3(x), *_strides3(W), *_strides3(gu),
        BM=BM, BN=BN, BK=BK, num_warps=w, num_stages=s)
    return gu


def swiglu_elem(gu, out, inter, cfg):
    BM, BN, w = cfg
    E = gu.shape[0] if gu.dim() == 3 else 1
    M = gu.shape[-2]
    _swiglu_elem_kernel[(triton.cdiv(M, BM), triton.cdiv(inter, BN), E)](
        gu, out, M, inter, *_strides3(gu), *_strides3(out), BM=BM, BN=BN, num_warps=w)
    return out


def swiglu_fused(x, W, out, inter, cfg):
    BM, BN, BK, w, s = cfg
    E = x.shape[0] if x.dim() == 3 else 1
    M, K = x.shape[-2], x.shape[-1]
    _swiglu_fused_kernel[(triton.cdiv(M, BM), triton.cdiv(inter, BN), E)](
        x, W, out, M, inter, K, *_strides3(x), *_strides3(W), *_strides3(out),
        BM=BM, BN=BN, BK=BK, num_warps=w, num_stages=s)
    return out


def tune(make_call, cands):
    """(best_seconds, best_cfg) over candidates; compile/launch failures skipped."""
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
def measure_config(name, M, hidden, inter, count, group, iters, warmup, gpu, gpu_adj,
                   smoke=False):
    """One N4 config, custom-vs-custom. count=1 dense; count=E grouped (batched tensors)."""
    batched = count > 1
    print(f"[n4_custom] {name}: M={M} hidden={hidden} inter={inter} count={count} "
          f"({group}) ...", flush=True)
    if batched:
        x = bf(count, M, hidden)
        W = bf(count, hidden, 2 * inter)
        gu = torch.empty((count, M, 2 * inter), device="cuda:0", dtype=torch.bfloat16)
        act = torch.empty((count, M, inter), device="cuda:0", dtype=torch.bfloat16)
    else:
        x = bf(M, hidden)
        W = bf(hidden, 2 * inter)
        gu = torch.empty((M, 2 * inter), device="cuda:0", dtype=torch.bfloat16)
        act = torch.empty((M, inter), device="cuda:0", dtype=torch.bfloat16)

    # fp32 reference + eager context
    xf, Wf = x.float(), W.float()
    gu32 = torch.matmul(xf, Wf)
    ref32 = F.silu(gu32[..., :inter]) * gu32[..., inter:]
    del gu32, xf, Wf
    eager_out = F.silu((torch.matmul(x, W))[..., :inter]) * (torch.matmul(x, W))[..., inter:]
    eager_rel = _rel_max(eager_out, ref32)
    rel_tol = max(2.0 * eager_rel, 5e-2)
    del eager_out

    cands = GEMM_CANDS if not smoke else GEMM_CANDS[:3]
    ecands = ELEM_CANDS if not smoke else ELEM_CANDS[:2]
    row = {"name": name, "kind": "n4_custom", "group": group,
           "dims": {"M": M, "hidden": hidden, "inter": inter, "count": count,
                    "batched": batched}}
    with ClockSampler() as cs:
        # vendor context (NOT the comparison target)
        vendor_gemm_s = med_time(lambda: torch.matmul(x, W), iters, warmup)
        vendor_unfused_s = med_time(
            lambda: F.silu((torch.matmul(x, W))[..., :inter])
            * (torch.matmul(x, W))[..., inter:], iters, warmup)

        # --- custom UNFUSED: tune each stage, then time the 2-kernel sequence ---
        tg = tune(lambda c: (lambda: gemm_full(x, W, gu, c)), cands)
        te = tune(lambda c: (lambda: swiglu_elem(gu, act, inter, c)), ecands)
        if tg is None or te is None:
            raise RuntimeError(f"no unfused candidate compiled (gemm={tg}, elem={te})")
        g_cfg, e_cfg = tg[1], te[1]
        unf_seq_s = med_time(
            lambda: (gemm_full(x, W, gu, g_cfg), swiglu_elem(gu, act, inter, e_cfg)),
            iters, warmup)
        gemm_full(x, W, gu, g_cfg)
        swiglu_elem(gu, act, inter, e_cfg)
        unf_rel = _rel_max(act, ref32)

        # --- custom FUSED: tune, time ---
        tf = tune(lambda c: (lambda: swiglu_fused(x, W, act, inter, c)), cands)
        if tf is None:
            raise RuntimeError("no fused candidate compiled")
        f_cfg = tf[1]
        fus_s = med_time(lambda: swiglu_fused(x, W, act, inter, f_cfg), iters, warmup)
        swiglu_fused(x, W, act, inter, f_cfg)
        fus_rel = _rel_max(act, ref32)

        # --- same-tile isolation: identical (BM,BN,BK,w,s) on gemm_full and fused ---
        st_cfg = f_cfg
        try:
            unf_st_s = med_time(
                lambda: (gemm_full(x, W, gu, st_cfg), swiglu_elem(gu, act, inter, e_cfg)),
                iters, warmup)
            fus_st_s = fus_s  # fused already timed at st_cfg == its best cfg
            same_tile_gain = unf_st_s / fus_st_s
        except Exception as exc:
            unf_st_s, same_tile_gain = None, None
            row["same_tile_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"

        # drift probe: custom GEMM re-measured at config end
        gemm_repeat_s = med_time(lambda: gemm_full(x, W, gu, g_cfg), iters, warmup)
        gemm_first_s = tg[0]
    clocks = cs.summary()
    sleep_cooldown()

    unf_ok = bool(unf_rel <= rel_tol)
    fus_ok = bool(fus_rel <= rel_tol)
    est_unf, est_fus = est_swiglu_ms(M, hidden, inter, count, gpu)
    row.update({
        "custom_unfused_ms": unf_seq_s * 1e3,
        "custom_unfused_cfg": {"gemm": g_cfg, "elem": e_cfg},
        "custom_fused_ms": fus_s * 1e3,
        "custom_fused_cfg": f_cfg,
        "custom_gain_best": unf_seq_s / fus_s,
        "custom_unfused_same_tile_ms": unf_st_s and unf_st_s * 1e3,
        "custom_gain_same_tile": same_tile_gain,
        "vendor_gemm_only_ms": vendor_gemm_s * 1e3,
        "vendor_eager_unfused_ms": vendor_unfused_s * 1e3,
        "custom_gemm_over_vendor_gemm": (tg[0] / vendor_gemm_s),
        "numerics": {"eager_rel_max_vs_fp32": eager_rel, "rel_tol": rel_tol,
                     "unfused_rel_max_vs_fp32": unf_rel, "unfused_ok": unf_ok,
                     "fused_rel_max_vs_fp32": fus_rel, "fused_ok": fus_ok},
        "est_unfused_ms": est_unf, "est_fused_ms": est_fus,
        "estimated_gain": est_unf / est_fus,
        "gemm_drift_ratio": gemm_repeat_s / gemm_first_s,
        "clocks": clocks,
    })
    if gpu_adj is not None:
        a_unf, a_fus = est_swiglu_ms(M, hidden, inter, count, gpu_adj)
        row["est_unfused_ms_adj"], row["est_fused_ms_adj"] = a_unf, a_fus
        row["estimated_gain_adj"] = a_unf / a_fus
    del x, W, gu, act, ref32
    torch.cuda.empty_cache()
    print(f"    custom: unf={row['custom_unfused_ms']:.4f} fus={row['custom_fused_ms']:.4f} "
          f"gain_best={row['custom_gain_best']:.4f} gain_same_tile={same_tile_gain} "
          f"est={row['estimated_gain']:.4f} num(unf/fus)={unf_ok}/{fus_ok} "
          f"drift={row['gemm_drift_ratio']:.3f} "
          f"[vendor ctx: gemm {row['vendor_gemm_only_ms']:.4f}, "
          f"eager {row['vendor_eager_unfused_ms']:.4f}]", flush=True)
    return row


def configs(smoke):
    if smoke:
        return [("n4c_smoke_dense", 512, 512, 512, 1, "t4_dense"),
                ("n4c_smoke_moe", 64, 512, 512, 4, "t4_moe")]
    cfgs = []
    for M in (2048, 8192):
        for h in (1024, 2048, 4096):
            cfgs.append((f"n4c_t4_M{M}_h{h}", M, h, h, 1, "t4_dense"))
    for E in (8, 32):
        cfgs.append((f"n4c_t4_moe_E{E}", 128, 2048, 2048, E, "t4_moe"))
    for tpe in (16, 32, 64, 128, 256, 512, 1024, 4096):
        cfgs.append((f"n4c_glm_tpe{tpe}", tpe, 6144, 2048, 1, "glm_dense"))
    for tpe in (64, 512):
        cfgs.append((f"n4c_glm_grouped_tpe{tpe}", tpe, 6144, 2048, 8, "glm_grouped"))
    return cfgs


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
        "task": "N4 addendum: custom-fused vs custom-unfused (host request, round 4)",
        "torch": torch.__version__, "triton": triton.__version__,
        "device": torch.cuda.get_device_name(0),
        "clocks_locked": True,
        "clocks_note": "clocks LOCKED 1500/5501 by host; ClockSampler per config",
        "smoke": args.smoke, "adjusted_profile": adj_info,
    }
    conventions = {
        "comparison": "custom_gain_* compares the custom FUSED kernel against a fully-custom "
                      "2-kernel UNFUSED implementation of the same Triton tiling family (plain "
                      "full-width GEMM + separate SwiGLU elementwise). Vendor numbers "
                      "(cuBLAS gemm_only, eager unfused) are CONTEXT columns only.",
        "custom_gain_best": "best-vs-best: each side independently tile-tuned",
        "custom_gain_same_tile": "unfused GEMM forced to the fused kernel's best "
                                 "(BM,BN,BK,warps,stages) -- pure mechanism isolation",
        "estimator": "est_swiglu_ms assumes the SAME optimal-mapping GEMM on both sides -- "
                     "custom-vs-custom is exactly the estimator's regime",
        "units": "ms for *_ms; gain = unfused/fused (>1 => fusion faster)",
        "timing": f"median of {iters} cuda events after {warmup} warmup + clock warmer",
        "numerics": "rel-vs-fp32 <= max(2*eager_rel, 5e-2)",
    }
    results = []
    out = {"conventions": conventions, "env": env, "configs": results}
    for name, M, h, inter, count, group in configs(args.smoke):
        try:
            row = measure_config(name, M, h, inter, count, group, iters, warmup,
                                 gpu, gpu_adj, smoke=args.smoke)
        except Exception as exc:
            traceback.print_exc()
            row = {"name": name, "kind": "n4_custom", "group": group,
                   "error": f"{type(exc).__name__}: {exc}"}
        results.append(row)
        save_json(args.out, out)

    ok = [r for r in results if "error" not in r]
    print("\n=== N4 custom-vs-custom summary ===")
    print(f"{'config':<26}{'group':<13}{'gain_best':>10}{'same_tile':>10}{'est':>8}"
          f"{'num':>6}{'drift':>7}")
    for r in ok:
        print(f"{r['name']:<26}{r['group']:<13}{r['custom_gain_best']:>10.4f}"
              f"{(r['custom_gain_same_tile'] or float('nan')):>10.4f}"
              f"{r['estimated_gain']:>8.4f}"
              f"{str(r['numerics']['fused_ok'])[:1]:>6}{r['gemm_drift_ratio']:>7.3f}")
    for g in ("t4_dense", "t4_moe", "glm_dense", "glm_grouped"):
        sel = [r for r in ok if r["group"] == g and r["numerics"]["fused_ok"]
               and r["numerics"]["unfused_ok"] and abs(r["gemm_drift_ratio"] - 1) <= 0.05]
        if sel:
            print(f"[{g}] clean n={len(sel)}: gain_best gm={geomean([r['custom_gain_best'] for r in sel]):.4f} "
                  f"same_tile gm={geomean([r['custom_gain_same_tile'] for r in sel if r['custom_gain_same_tile']]):.4f} "
                  f"est gm={geomean([r['estimated_gain'] for r in sel]):.4f}")


if __name__ == "__main__":
    main()
