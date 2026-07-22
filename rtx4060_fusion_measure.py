"""T4 (RTX4060_SIM_REAL_TASK.md) - fusion realizability: the KEY experiment.

Measures, on working NVIDIA/CUDA paths, whether the estimator's predicted ~1-2.6% fusion
gain for the GLM-5.2 decode fusions materialises on real silicon -- disambiguating the C500
null result (A: real-but-un-capturable-on-C500 tooling problem, vs B: estimator over-predicts).

Two fusion primitives, each on scaled dense configs that fit the 8 GiB 4060:
  PRIMITIVE 1  SwiGLU folded into the up_gate epilogue (F4-analog):
      swiglu_ffn(x, Wug): gu = x@Wug ; g,u = gu[..,:inter],gu[..,inter:] ; silu(g)*u
      paths: eager unfused | torch.compile(max-autotune) | (optional) hand Triton GEMM+SwiGLU
  PRIMITIVE 2  residual folded into the mla_o epilogue (F1-analog):
      (x@Wo)+res  vs  torch.addmm(res,x,Wo) (cuBLAS beta-accumulate)  vs  compile(max-autotune)

For every config the estimator's predicted gain (fusion_time_estimator, rtx4060-measured
profile, same dims) is computed CPU-side and stored next to the measured gain.

UNITS: rtx4060_common.med_time and the estimator KTime.time_s are both SECONDS; every field
ending _ms in the output JSON is milliseconds. gain = unfused/fused (>1 => fusion FASTER),
matching the estimator's "1.020x" convention.

Run (from GEMM-mapping/):
    python rtx4060_fusion_measure.py --out rtx4060_fusion.json
    python rtx4060_fusion_measure.py --smoke --out /tmp/.../fusion_smoke.json
    python rtx4060_fusion_measure.py --out rtx4060_fusion.json --moe          # + grouped-bmm MoE
    python rtx4060_fusion_measure.py --out rtx4060_fusion.json --t2-json rtx4060_measured.json

T6 extensions (RTX4060_SIM_REAL_TASK.md T6.C/D/E). Each new flag runs ONLY its new config
group (the T4 swiglu/residual/moe groups are skipped whenever any of them is passed), and
main() MERGE-APPENDS: if --out already exists its rows + annotations_post_hoc are preserved
and the new rows are appended (incremental re-save after every config):
    python rtx4060_fusion_measure.py --out rtx4060_fusion.json --topk  --t2-json rtx4060_peak.json
    python rtx4060_fusion_measure.py --out rtx4060_fusion.json --ffn   --t2-json rtx4060_peak.json
    python rtx4060_fusion_measure.py --out rtx4060_fusion.json --merge --t2-json rtx4060_peak.json
    python rtx4060_fusion_measure.py --smoke --topk --ffn --merge --out /tmp/.../cde_smoke.json
  --topk  T6.C router top-k(=8) as the router-GEMM epilogue (attempt-and-DROP protocol;
          rows kind="router_topk"; est cell is a structure-blind traffic-only bound).
  --ffn   T6.D MoE FFN fusion levels L1/L2/L3(F6) at GLM per-expert dims (rows
          kind="ffn_levels" + kind="ffn_grouped" for the D.7 8-expert cross-check). Runs the
          D.5 estimate_ffn_fused_m acceptance self-check FIRST and HALTS if it fails.
  --merge T6.E residual2 into the expert-merge reduction (rows kind="merge_r2f"; no GEMM --
          pure memory-bound; token-independence assertion covers the 131072 drop).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import traceback

import torch
import torch.nn.functional as F
import torch._dynamo as _dynamo
import torch._inductor.config as _ic
import torch._inductor.utils as _iu

# max-autotune lets inductor *try* to emit a Triton GEMM template with a fused epilogue
# (on <=24 SM parts it may decline -- we record whichever it chose).
_ic.max_autotune = True
_ic.max_autotune_gemm = True


class force_triton_templates:
    """Make inductor emit a Triton GEMM template on this 24-SM part.

    is_big_gpu() hard-requires >=68 SMs (torch/_inductor/utils.py), so max-autotune on the
    4060 keeps the vendor GEMM and only fuses the pointwise epilogue into a SEPARATE kernel.
    Restricting the autotune backends to TRITON (instead of letting cuBLAS win) is what
    yields the actual object under test: a Triton GEMM template with the epilogue folded in.
    The backend change also changes the inductor cache key, so forced and unforced variants
    can never reuse each other's compiled artifacts. Must wrap COMPILE + FIRST CALLS
    (lowering happens on first call, not at torch.compile())."""

    def __enter__(self):
        self._orig_big = _iu.is_big_gpu
        self._orig_backends = _ic.max_autotune_gemm_backends
        _iu.is_big_gpu = lambda *a, **k: True
        _ic.max_autotune_gemm_backends = "TRITON"
        return self

    def __exit__(self, *exc):
        _iu.is_big_gpu = self._orig_big
        _ic.max_autotune_gemm_backends = self._orig_backends

# The full sweep compiles up to 8 distinct shapes of the SAME swiglu_ffn code object
# (6 dense + 2 MoE) -- exactly torch's default recompile limit of 8, past which dynamo
# SILENTLY falls back to eager (compiled_ms would then time the eager path). Raise it.
# T6 sweeps many more shapes per code object (9 topk + 9+2 ffn + 8 merge) -> >= 256.
for _attr in ("cache_size_limit", "recompile_limit"):
    if getattr(_dynamo.config, _attr, 256) < 256:
        setattr(_dynamo.config, _attr, 256)

sys.path.insert(0, "/home/shuhan/snowcat-demo/GEMM-mapping")

from rtx4060_common import (  # noqa: E402  (mandatory shared timing module; med_time returns SECONDS)
    DEV,
    ClockSampler,
    bf,
    geomean,
    med_time,
    save_json,
    sleep_cooldown,
)
from gemm_time_estimator import GPUS  # noqa: E402
from fusion_time_estimator import (  # noqa: E402
    BPE,
    EXPERTS,
    HIDDEN,
    INTERMEDIATE,
    TOP_K,
    Epilogue,
    _residual_aux,
    estimate_ffn_fused_m,
    estimate_fused_gemm,
    estimate_gemm_grouped,
    estimate_vector_kernel,
)

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False


# --------------------------------------------------------------------------- #
# Hand Triton GEMM + SwiGLU epilogue (the "upper bound" / fusion-tax datapoint)  #
# --------------------------------------------------------------------------- #
if _HAVE_TRITON:

    @triton.jit
    def _swiglu_gemm_kernel(
        X, W, O,
        M, INTER, K,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_om, stride_on,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        # One output tile O[BM, BN] over the INTER (half-width) dimension needs the GEMM's
        # gate columns [n : n+BN] AND up columns [n+INTER : n+INTER+BN]; accumulate both.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        x_ptrs = X + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        wg_ptrs = W + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        wu_ptrs = W + (offs_k[:, None] * stride_wk + (offs_n[None, :] + INTER) * stride_wn)
        acc_g = tl.zeros((BM, BN), dtype=tl.float32)
        acc_u = tl.zeros((BM, BN), dtype=tl.float32)
        m_mask = offs_m[:, None] < M
        n_mask = offs_n[None, :] < INTER
        for k0 in range(0, K, BK):
            kmask = offs_k < (K - k0)
            a = tl.load(x_ptrs, mask=m_mask & kmask[None, :], other=0.0)
            wg = tl.load(wg_ptrs, mask=kmask[:, None] & n_mask, other=0.0)
            wu = tl.load(wu_ptrs, mask=kmask[:, None] & n_mask, other=0.0)
            acc_g += tl.dot(a, wg)
            acc_u += tl.dot(a, wu)
            x_ptrs += BK * stride_xk
            wg_ptrs += BK * stride_wk
            wu_ptrs += BK * stride_wk
        out = (acc_g * tl.sigmoid(acc_g)) * acc_u   # silu(gate) * up
        o_ptrs = O + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
        tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=m_mask & n_mask)

    def triton_swiglu(x, Wug, inter, BM=64, BN=64, BK=32, num_warps=4, num_stages=2):
        M, K = x.shape
        O = torch.empty((M, inter), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(M, BM), triton.cdiv(inter, BN))
        _swiglu_gemm_kernel[grid](
            x, Wug, O, M, inter, K,
            x.stride(0), x.stride(1),
            Wug.stride(0), Wug.stride(1),
            O.stride(0), O.stride(1),
            BM=BM, BN=BN, BK=BK, num_warps=num_warps, num_stages=num_stages,
        )
        return O

    # (BM, BN, BK, num_warps, num_stages) candidates for a quick hand-rolled autotune.
    TRITON_CANDS = [
        (64, 64, 32, 4, 2), (128, 64, 32, 4, 3), (64, 128, 32, 4, 3),
        (128, 128, 32, 8, 3), (128, 128, 64, 8, 4), (64, 64, 64, 4, 3),
    ]

    def tune_triton_swiglu(x, Wug, inter, cands):
        """Pick the fastest (BM,BN,BK,w,s) with a short timing pass; returns (cfg, none-if-all-fail)."""
        best = None
        for cfg in cands:
            BM, BN, BK, w, s = cfg
            try:
                t = med_time(lambda: triton_swiglu(x, Wug, inter, BM, BN, BK, w, s),
                             iters=5, warmup=3)
            except Exception:
                continue
            if best is None or t < best[0]:
                best = (t, cfg)
        return best[1] if best else None

    # ----------------------------------------------------------------------- #
    # T6.C -- standalone row-resident top-k (baseline: not penalized by torch. #
    # topk's generic radix-select; reads logits ONCE) and the one viable fused #
    # route: BN=N=256 full-row GEMM + 8x iterative argmax-and-mask epilogue.   #
    # ----------------------------------------------------------------------- #
    @triton.jit
    def _row_topk_kernel(L, VALS, IDX, M, stride_lm,
                         NEXP: tl.constexpr, KSEL: tl.constexpr, BM: tl.constexpr):
        pid = tl.program_id(0)
        offs_m = pid * BM + tl.arange(0, BM)
        offs_n = tl.arange(0, NEXP)
        row_mask = offs_m < M
        x = tl.load(L + offs_m[:, None] * stride_lm + offs_n[None, :],
                    mask=row_mask[:, None], other=float("-inf")).to(tl.float32)
        for j in tl.static_range(KSEL):
            v = tl.max(x, axis=1)
            i = tl.argmax(x, axis=1)
            tl.store(VALS + offs_m * KSEL + j, v.to(VALS.dtype.element_ty), mask=row_mask)
            tl.store(IDX + offs_m * KSEL + j, i.to(tl.int32), mask=row_mask)
            x = tl.where(offs_n[None, :] == i[:, None], float("-inf"), x)

    def triton_row_topk(logits, BM=64, num_warps=4, ksel=8):
        assert logits.stride(1) == 1
        M, NEXP = logits.shape
        vals = torch.empty((M, ksel), device=logits.device, dtype=logits.dtype)
        idx = torch.empty((M, ksel), device=logits.device, dtype=torch.int32)
        _row_topk_kernel[(triton.cdiv(M, BM),)](
            logits, vals, idx, M, logits.stride(0),
            NEXP=NEXP, KSEL=ksel, BM=BM, num_warps=num_warps)
        return vals, idx

    ROWTOPK_CANDS = [(64, 4), (128, 4), (32, 4), (128, 8)]      # (BM, num_warps)

    @triton.jit
    def _router_topk_gemm_kernel(
        X, W, VALS, IDX, M, K,
        stride_xm, stride_xk, stride_wk, stride_wn,
        NEXP: tl.constexpr, KSEL: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
    ):
        # One CTA owns the FULL logits row-block [BM, NEXP=256] (no N-tiling -- top-k is a
        # full-row selection), then does KSEL=8 iterative argmax-and-mask over the fp32
        # accumulator and writes vals[BM,8] + idx[BM,8] (never the 256 logits).
        pid = tl.program_id(0)
        offs_m = pid * BM + tl.arange(0, BM)
        offs_n = tl.arange(0, NEXP)
        offs_k = tl.arange(0, BK)
        row_mask = offs_m < M
        acc = tl.zeros((BM, NEXP), dtype=tl.float32)
        x_ptrs = X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = W + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        for k0 in range(0, K, BK):
            kmask = offs_k < (K - k0)
            a = tl.load(x_ptrs, mask=row_mask[:, None] & kmask[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=kmask[:, None], other=0.0)
            acc += tl.dot(a, w)
            x_ptrs += BK * stride_xk
            w_ptrs += BK * stride_wk
        acc = tl.where(row_mask[:, None], acc, float("-inf"))
        for j in tl.static_range(KSEL):
            v = tl.max(acc, axis=1)
            i = tl.argmax(acc, axis=1)
            tl.store(VALS + offs_m * KSEL + j, v.to(VALS.dtype.element_ty), mask=row_mask)
            tl.store(IDX + offs_m * KSEL + j, i.to(tl.int32), mask=row_mask)
            acc = tl.where(offs_n[None, :] == i[:, None], float("-inf"), acc)

    def triton_router_topk(x, W, BM=32, BK=64, num_warps=4, num_stages=2, ksel=8):
        M, K = x.shape
        NEXP = W.shape[1]
        vals = torch.empty((M, ksel), device=x.device, dtype=x.dtype)
        idx = torch.empty((M, ksel), device=x.device, dtype=torch.int32)
        _router_topk_gemm_kernel[(triton.cdiv(M, BM),)](
            x, W, vals, idx, M, K,
            x.stride(0), x.stride(1), W.stride(0), W.stride(1),
            NEXP=NEXP, KSEL=ksel, BM=BM, BK=BK,
            num_warps=num_warps, num_stages=num_stages)
        return vals, idx

    # C.4 autotune space: BM {16,32,64} x BK {32,64} x warps {4,8} x stages {1,2}
    ROUTER_TOPK_CANDS = [(BM, BK, w, s)
                         for BM in (16, 32, 64) for BK in (32, 64)
                         for w in (4, 8) for s in (1, 2)]

    # ----------------------------------------------------------------------- #
    # T6.D -- F6 hand kernel: up_gate -> SwiGLU -> down as ONE kernel.          #
    # grid (E*mt,), BM=16 hard cap (D.4 SMEM table); the activated [16, INTER] #
    # block is kept on-chip as FOUR register-resident chunks of INTER/4 cols   #
    # (Triton has no dynamically-indexable register tensors, so BN_ug=INTER/4  #
    # and BK_d=INTER/4 are fixed by this static-chunk formulation).            #
    # ----------------------------------------------------------------------- #
    @triton.jit
    def _f6_phase1_chunk(Xb, Ub, offs_m, n_base, M,
                         stride_xm, stride_xk, stride_uk, stride_un,
                         HID: tl.constexpr, INTER: tl.constexpr, CH: tl.constexpr,
                         BM: tl.constexpr, BK: tl.constexpr):
        # up_gate + SwiGLU for inter-columns [n_base, n_base+CH): dual fp32 accumulators
        # over the full K=HID loop, silu(gate)*up applied on-chip, returned as bf16.
        offs_n = n_base + tl.arange(0, CH)
        offs_k = tl.arange(0, BK)
        row_mask = offs_m[:, None] < M
        acc_g = tl.zeros((BM, CH), dtype=tl.float32)
        acc_u = tl.zeros((BM, CH), dtype=tl.float32)
        for k0 in range(0, HID, BK):          # HID % BK == 0 (asserted host-side)
            a = tl.load(Xb + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xk,
                        mask=row_mask, other=0.0)
            wg = tl.load(Ub + (k0 + offs_k)[:, None] * stride_uk + offs_n[None, :] * stride_un)
            wu = tl.load(Ub + (k0 + offs_k)[:, None] * stride_uk
                         + (offs_n + INTER)[None, :] * stride_un)
            acc_g += tl.dot(a, wg)
            acc_u += tl.dot(a, wu)
        return ((acc_g * tl.sigmoid(acc_g)) * acc_u).to(tl.bfloat16)

    @triton.jit
    def _f6_ffn_kernel(X, WUG, WDN, O, M,
                       stride_xe, stride_xm, stride_xk,
                       stride_ue, stride_uk, stride_un,
                       stride_de, stride_dk, stride_dn,
                       stride_oe, stride_om, stride_on,
                       MT,
                       HID: tl.constexpr, INTER: tl.constexpr, CH: tl.constexpr,
                       BM: tl.constexpr, BK_UG: tl.constexpr, BN_D: tl.constexpr):
        pid = tl.program_id(0)
        e = pid // MT                      # expert (grouped D.7 grid = (8*mt,); dense e=0)
        rb = pid % MT                      # row-block within the expert
        offs_m = rb * BM + tl.arange(0, BM)
        Xb = X + e * stride_xe
        Ub = WUG + e * stride_ue
        Db = WDN + e * stride_de
        Ob = O + e * stride_oe
        # phase 1: fill the activated [BM, INTER] block (four CH-wide chunks)
        act0 = _f6_phase1_chunk(Xb, Ub, offs_m, 0 * CH, M, stride_xm, stride_xk,
                                stride_uk, stride_un, HID, INTER, CH, BM, BK_UG)
        act1 = _f6_phase1_chunk(Xb, Ub, offs_m, 1 * CH, M, stride_xm, stride_xk,
                                stride_uk, stride_un, HID, INTER, CH, BM, BK_UG)
        act2 = _f6_phase1_chunk(Xb, Ub, offs_m, 2 * CH, M, stride_xm, stride_xk,
                                stride_uk, stride_un, HID, INTER, CH, BM, BK_UG)
        act3 = _f6_phase1_chunk(Xb, Ub, offs_m, 3 * CH, M, stride_xm, stride_xk,
                                stride_uk, stride_un, HID, INTER, CH, BM, BK_UG)
        # phase 2: down GEMM streams Wdn once against the resident activated block
        offs_kd = tl.arange(0, CH)
        row_mask = offs_m[:, None] < M
        for n0 in range(0, HID, BN_D):     # HID % BN_D == 0 (asserted host-side)
            offs_n = n0 + tl.arange(0, BN_D)
            acc = tl.dot(act0, tl.load(Db + (0 * CH + offs_kd)[:, None] * stride_dk
                                       + offs_n[None, :] * stride_dn))
            acc += tl.dot(act1, tl.load(Db + (1 * CH + offs_kd)[:, None] * stride_dk
                                        + offs_n[None, :] * stride_dn))
            acc += tl.dot(act2, tl.load(Db + (2 * CH + offs_kd)[:, None] * stride_dk
                                        + offs_n[None, :] * stride_dn))
            acc += tl.dot(act3, tl.load(Db + (3 * CH + offs_kd)[:, None] * stride_dk
                                        + offs_n[None, :] * stride_dn))
            tl.store(Ob + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
                     acc.to(O.dtype.element_ty), mask=row_mask)

    def triton_f6_ffn(x, Wug, Wdn, BK_UG=64, BN_D=64, num_warps=8, num_stages=2):
        """x [M,hid] or [E,M,hid]; Wug [(E,)hid,2*inter]; Wdn [(E,)inter,hid] -> out [(E,)M,hid]."""
        batched = x.dim() == 3
        if batched:
            E, M, hid = x.shape
        else:
            (M, hid), E = x.shape, 1
        inter = Wdn.shape[-2]
        assert Wug.shape[-1] == 2 * inter and Wdn.shape[-1] == hid
        assert hid % BK_UG == 0 and hid % BN_D == 0 and inter % 4 == 0 and M % 16 == 0
        MT = M // 16
        if batched:
            O = torch.empty((E, M, hid), device=x.device, dtype=x.dtype)
            sxe, sxm, sxk = x.stride(0), x.stride(1), x.stride(2)
            sue, suk, sun = Wug.stride(0), Wug.stride(1), Wug.stride(2)
            sde, sdk, sdn = Wdn.stride(0), Wdn.stride(1), Wdn.stride(2)
            soe, som, son = O.stride(0), O.stride(1), O.stride(2)
        else:
            O = torch.empty((M, hid), device=x.device, dtype=x.dtype)
            sxe, sxm, sxk = 0, x.stride(0), x.stride(1)
            sue, suk, sun = 0, Wug.stride(0), Wug.stride(1)
            sde, sdk, sdn = 0, Wdn.stride(0), Wdn.stride(1)
            soe, som, son = 0, O.stride(0), O.stride(1)
        _f6_ffn_kernel[(E * MT,)](
            x, Wug, Wdn, O, M,
            sxe, sxm, sxk, sue, suk, sun, sde, sdk, sdn, soe, som, son,
            MT, HID=hid, INTER=inter, CH=inter // 4,
            BM=16, BK_UG=BK_UG, BN_D=BN_D,
            num_warps=num_warps, num_stages=num_stages)
        return O

    # (BK_UG, BN_D, num_warps, num_stages) -- modest autotune per D.4 (BM=16 hard cap;
    # BN_ug = BK_d = INTER/4 are fixed by the static-chunk formulation, see kernel note)
    F6_CANDS = [(64, 64, 8, 2), (64, 128, 8, 2), (128, 64, 8, 2),
                (64, 64, 4, 2), (64, 64, 8, 1), (128, 128, 8, 2)]

    # ----------------------------------------------------------------------- #
    # T6.E -- hand merge+residual2 kernel (E.4 pseudocode; memory-peak bound).  #
    # ----------------------------------------------------------------------- #
    @triton.jit
    def _merge_r2f_kernel(EO, G, R, O, T, H,
                          stride_et, stride_ee,
                          KE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rows = pid_m * BM + tl.arange(0, BM)
        cols = pid_n * BN + tl.arange(0, BN)
        m = rows[:, None] < T
        n = cols[None, :] < H
        # int64 row offsets: expert_outs flat indices reach T*8*H (2.42e9 > 2^31 at
        # T=49152) -- int32 pointer arithmetic overflows to an illegal address there
        rows64 = rows.to(tl.int64)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for e in tl.static_range(KE):                            # TOP_K=8, unrolled
            g = tl.load(G + rows * KE + e, mask=rows < T, other=0.0).to(tl.float32)
            x = tl.load(EO + rows64[:, None] * stride_et + e * stride_ee + cols[None, :],
                        mask=m & n, other=0.0).to(tl.float32)
            acc += g[:, None] * x
        acc += tl.load(R + rows64[:, None] * H + cols[None, :],  # tile-local residual2 add
                       mask=m & n, other=0.0).to(tl.float32)
        tl.store(O + rows64[:, None] * H + cols[None, :], acc.to(O.dtype.element_ty),
                 mask=m & n)

    def triton_merge_r2f(eo, g, r2, BM=32, BN=128, num_warps=4):
        T, KE, H = eo.shape
        assert eo.stride(2) == 1 and r2.stride(1) == 1 and g.stride(1) == 1
        O = torch.empty((T, H), device=eo.device, dtype=eo.dtype)
        _merge_r2f_kernel[(triton.cdiv(T, BM), triton.cdiv(H, BN))](
            eo, g, r2, O, T, H, eo.stride(0), eo.stride(1),
            KE=KE, BM=BM, BN=BN, num_warps=num_warps)
        return O

    MERGE_CANDS = [(32, 128, 4), (64, 64, 4), (16, 256, 4),
                   (32, 256, 8), (64, 128, 8), (128, 64, 4)]    # (BM, BN, num_warps)


def _quick_tune(make_call, cands, iters=5, warmup=3):
    """Mini-autotune shared by the T6 hand kernels: shortest med_time over `cands`;
    per-candidate compile/launch failures are caught and skipped (C.4/D.4/E.4).
    Returns (best_seconds, best_cfg) or None if every candidate failed."""
    best = None
    for cfg in cands:
        try:
            t = med_time(make_call(cfg), iters=iters, warmup=warmup)
        except Exception:
            continue
        if best is None or t < best[0]:
            best = (t, cfg)
    return best


# --------------------------------------------------------------------------- #
# Fusion verification helpers (profiler + numerics)                             #
# --------------------------------------------------------------------------- #
def profile_kernels(call):
    """Return the CUDA kernels (name, us) launched by one call() invocation.

    activities=[ProfilerActivity.CUDA] per task spec; CPU-side ops carry dt=0 and are
    filtered out via device_type==CUDA.
    """
    from torch.profiler import profile, ProfilerActivity
    from torch.autograd import DeviceType

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    kernels = []
    for ev in prof.key_averages():
        if ev.device_type != DeviceType.CUDA:
            continue
        dt = getattr(ev, "device_time_total", None)
        if dt is None:
            dt = getattr(ev, "cuda_time_total", 0.0)
        dt = float(dt or 0.0)
        if dt <= 0:
            continue
        cnt = int(getattr(ev, "count", 1) or 1)
        kernels.append({
            "name": ev.key,
            "count": cnt,
            "total_us": round(dt, 3),
            "avg_us": round(dt / max(cnt, 1), 3),
        })
    kernels.sort(key=lambda d: -d["total_us"])
    return kernels


def _is_gemm_template(name_lower):
    # An inductor Triton GEMM *template* fuses the epilogue into the GEMM itself.
    return "triton" in name_lower and any(t in name_lower for t in ("mm", "gemm", "tem"))


def judge_fused(kernels, act_markers):
    """fused_verified iff a Triton GEMM template exists AND no separate activation kernel
    (silu/mul/add elementwise) consuming the intermediate remains. A tiny copy/reduction is
    tolerable (it just won't be the dominant kernel). Raw list is stored for human re-judge.
    """
    lows = [k["name"].lower() for k in kernels]
    total = sum(k["total_us"] for k in kernels) or 1.0
    has_template = any(_is_gemm_template(l) for l in lows)
    dominant = max(kernels, key=lambda k: k["total_us"]) if kernels else None
    dom_is_template = dominant is not None and _is_gemm_template(dominant["name"].lower())
    # separate elementwise activation/epilogue kernels that are NOT a GEMM template
    separate = [
        k["name"] for k, l in zip(kernels, lows)
        if not _is_gemm_template(l) and any(m in l for m in act_markers)
    ]
    fused = has_template and dom_is_template and len(separate) == 0
    evidence = {
        "n_kernels": len(kernels),
        "dominant_kernel": dominant["name"] if dominant else None,
        "dominant_frac_of_gpu_time": round(dominant["total_us"] / total, 4) if dominant else None,
        "triton_gemm_template_present": has_template,
        "separate_elementwise_kernels": separate,
    }
    return fused, evidence


def _maxabs(a, b):
    return (a.float() - b.float()).abs().max().item()


def _rel_max(t, ref32):
    """Max elementwise relative deviation from an fp32 reference (1.0 absolute floor)."""
    return ((t.float() - ref32).abs() / (ref32.abs() + 1.0)).max().item()


def vendor_fused_ok(kernels, markers):
    """Vendor-path fusion judge (no Triton template required): fused iff NO separate
    non-GEMM elementwise kernel matching `markers` remains. Memcpy DtoD is tolerated
    (for addmm it is the algorithmic beta-accumulate copy of `res`, not epilogue work)."""
    lows = [k["name"].lower() for k in kernels]
    separate = [
        k["name"] for k, l in zip(kernels, lows)
        if any(m in l for m in markers)
        and not _is_gemm_template(l) and "gemm" not in l and "cutlass" not in l
    ]
    return len(separate) == 0, separate


def _cudagraph_copy_us(kernels):
    """GPU time of the cudagraph static-input copy kernels (multi_tensor_apply) inside one
    compiled call. This is deployment overhead of the cudagraph path, NOT fusion work --
    at full dims it is ~2-3% of compiled_ms, the same order as the 1-2.6% gain under test,
    so it is stored separately for the writeup. (Memcpy DtoD is deliberately NOT counted:
    in the residual case it is the algorithmic beta-accumulate copy of `res`.)"""
    return sum(k["total_us"] for k in kernels if "multi_tensor_apply" in k["name"].lower())


# --------------------------------------------------------------------------- #
# Estimator side (CPU only -- no GPU)                                            #
# --------------------------------------------------------------------------- #
def est_swiglu_ms(M, hidden, inter, count, gpu):
    """F4-analog predicted (unfused_ms, fused_ms) for dense/grouped SwiGLU-into-up_gate.

    up_gate GEMM is (m=M, n=2*inter, k=hidden). count=1 dense; count=E grouped MoE.
    unfused = grouped GEMM + separate activation vector kernel (read gate+up 2*inter, write
    activated inter -> 3*rows*inter*BPE); fused = same GEMM with out_factor=0.5 epilogue.
    """
    rows = count * M
    unf_g = estimate_gemm_grouped("up_gate", M, 2 * inter, hidden, count, gpu)
    unf_a = estimate_vector_kernel("activation", 3 * rows * inter * BPE, gpu)
    fus = estimate_fused_gemm("up_gate+swiglu", M, 2 * inter, hidden, count,
                              Epilogue(out_factor=0.5), gpu)
    return (unf_g.time_s + unf_a.time_s) * 1e3, fus.time_s * 1e3


def est_residual_ms(M, n, k, gpu):
    """F1-analog predicted (unfused_ms, fused_ms) for residual-into-mla_o (dense, count=1).

    GEMM is (m=M, n, k). unfused = GEMM + residual vector kernel (read y, read x, write sum =
    3*M*n*BPE); fused = GEMM whose epilogue adds one residual read (extra_hbm_once=M*n*BPE) and
    holds the residual output tile on chip (aux SMEM = _residual_aux).
    """
    unf_g = estimate_gemm_grouped("gemm", M, n, k, 1, gpu)
    unf_r = estimate_vector_kernel("residual", 3 * M * n * BPE, gpu)
    fus = estimate_fused_gemm("gemm+residual", M, n, k, 1,
                              Epilogue(extra_hbm_once=M * n * BPE,
                                       aux_smem_per_tile=_residual_aux), gpu)
    return (unf_g.time_s + unf_r.time_s) * 1e3, fus.time_s * 1e3


def est_router_topk_ms(M, n, k, gpu):
    """T6.C.5 (unfused_ms, fused_ms). STRUCTURE-BLIND TRAFFIC-ONLY BOUND, NOT a prediction:
    the estimator has no top-k primitive (no selection compute, no BN=256 register/occupancy
    tax) -- rows carry est_traffic_only_bound=True and are EXCLUDED from aggregates."""
    unf = (estimate_gemm_grouped("router", M, n, k, 1, gpu).time_s
           + estimate_vector_kernel("topk_select", M * n * BPE + M * 8 * 6, gpu).time_s)
    fus = estimate_fused_gemm(
        "router+topk", M, n, k, 1,
        Epilogue(out_factor=(8 * (BPE + 4)) / (n * BPE),   # write 8 vals + 8 int32 idx, not n logits
                 extra_hbm_once=M * 8 * 4,                 # index-write extra HBM (8 int32/row)
                 aux_smem_per_tile=lambda m0, n0: m0 * n * 4),  # fp32 [m0, n] row accumulator
        gpu).time_s
    return unf * 1e3, fus * 1e3


def est_ffn_levels_ms(tpe, hidden, inter, gpu):
    """T6.D estimator trio for ONE expert (count=1), per-tpe: (est_L1, est_L2, est_L3, m0, mt)
    in ms. L3 comes from the tpe-parametrized estimate_ffn_fused_m (D.5 HARD prerequisite);
    it is GLM-dims-only (the helper reads HIDDEN/INTERMEDIATE internally)."""
    down = estimate_gemm_grouped("down", tpe, hidden, inter, 1, gpu).time_s
    l1 = (estimate_gemm_grouped("up_gate", tpe, 2 * inter, hidden, 1, gpu).time_s
          + estimate_vector_kernel("activation", 3 * tpe * inter * BPE, gpu).time_s
          + down)
    l2 = (estimate_fused_gemm("up_gate+swiglu", tpe, 2 * inter, hidden, 1,
                              Epilogue(out_factor=0.5), gpu).time_s
          + down)
    l3_ms = m0 = mt = None
    if (hidden, inter) == (HIDDEN, INTERMEDIATE):
        r = estimate_ffn_fused_m(tpe, 1, gpu)
        if r is not None:
            l3_ms, m0, mt = r[0] * 1e3, r[1], r[2]
    return l1 * 1e3, l2 * 1e3, l3_ms, m0, mt


def est_merge_r2f_ms(T, H, gpu):
    """T6.E.5 (unfused_ms, fused_ms). Pure memory-bound: unfused 12*T*H*BPE vs fused
    10*T*H*BPE -> estimated_gain = 12/10 = 1.20 exactly, PROFILE- and TOKEN-independent
    (bandwidth cancels in the ratio -> stock == T2-adjusted; T cancels -> flat over tokens)."""
    unf_merge = estimate_vector_kernel("merge", (8 * T * H + T * H) * BPE, gpu)       # 9*T*H
    unf_add = estimate_vector_kernel("residual2", (3 * T * H) * BPE, gpu)             # 3*T*H
    fused = estimate_vector_kernel("merge+res2", (8 * T * H + T * H + T * H) * BPE, gpu)  # 10*T*H
    return (unf_merge.time_s + unf_add.time_s) * 1e3, fused.time_s * 1e3


def _est_ffn_unfused_grouped_s(m, count, gpu):
    """Estimator L1 (up_gate GEMM + activation vector + down GEMM) over `count` experts, SECONDS."""
    return (estimate_gemm_grouped("up_gate", m, 2 * INTERMEDIATE, HIDDEN, count, gpu).time_s
            + estimate_vector_kernel("activation", count * m * 3 * INTERMEDIATE * BPE, gpu).time_s
            + estimate_gemm_grouped("down", m, HIDDEN, INTERMEDIATE, count, gpu).time_s)


def f6_estimator_acceptance(gpu):
    """T6.D.5 acceptance for estimate_ffn_fused_m -- MUST pass before the D sweep is trusted.

    (1) at m=64/count=EXPERTS it reproduces run()'s known F6 ratio ~0.259x against the L1
        estimate; (2) m=16 vs m=512 scale by ~the mt weight-reread factor (32x), not flat at
        the old m=64 global; (3) the enumerated m0 is capped at 16 by SMEM.
    Prints the D.5-style predicted table. Returns (ok, report_dict)."""
    checks = {}
    f64 = estimate_ffn_fused_m(64, EXPERTS, gpu)
    r64 = _est_ffn_unfused_grouped_s(64, EXPERTS, gpu) / f64[0]
    checks["c1_reproduces_F6_0p259_at_m64"] = {
        "ratio": r64, "expected": 0.259, "ok": bool(abs(r64 - 0.259) <= 0.005)}
    f16 = estimate_ffn_fused_m(16, EXPERTS, gpu)
    f512 = estimate_ffn_fused_m(512, EXPERTS, gpu)
    scale = f512[0] / f16[0]
    mt_factor = f512[2] / f16[2]
    checks["c2_scales_with_mt_not_flat"] = {
        "t_m512_over_t_m16": scale, "mt_factor": mt_factor,
        "ok": bool(0.5 * mt_factor <= scale <= 1.5 * mt_factor)}
    m0s = {m: estimate_ffn_fused_m(m, EXPERTS, gpu)[1] for m in (16, 64, 512, 4096)}
    checks["c3_m0_capped_at_16_by_smem"] = {
        "m0_by_m": m0s, "ok": bool(all(v == 16 for v in m0s.values()))}
    ok = all(c["ok"] for c in checks.values())

    table = []
    print("\n[D.5 acceptance] predicted F6/L2 table (estimator, count=256 -- NOT measurements):")
    print(f"{'tpe':>6}{'mt':>5}{'iso_L2':>9}{'r_L2':>8}{'r_L3(F6)':>10}{'F6/expert ms':>14}{'F6 x256 ms':>13}")
    for tpe in (16, 32, 64, 128, 256, 512, 1024, 4096):
        iso_u, iso_f = est_swiglu_ms(tpe, HIDDEN, INTERMEDIATE, EXPERTS, gpu)
        unf_s = _est_ffn_unfused_grouped_s(tpe, EXPERTS, gpu)
        down_s = estimate_gemm_grouped("down", tpe, HIDDEN, INTERMEDIATE, EXPERTS, gpu).time_s
        l2_s = (estimate_fused_gemm("up_gate+swiglu", tpe, 2 * INTERMEDIATE, HIDDEN, EXPERTS,
                                    Epilogue(out_factor=0.5), gpu).time_s + down_s)
        f6 = estimate_ffn_fused_m(tpe, EXPERTS, gpu)
        row = {"tpe": tpe, "mt": f6[2], "iso_L2": iso_u / iso_f, "r_L2": unf_s / l2_s,
               "r_L3": unf_s / f6[0], "f6_per_expert_ms": f6[0] * 1e3 / EXPERTS,
               "f6_layer_ms": f6[0] * 1e3, "m0": f6[1]}
        table.append(row)
        print(f"{tpe:>6}{row['mt']:>5}{row['iso_L2']:>9.3f}{row['r_L2']:>8.3f}"
              f"{row['r_L3']:>10.3f}{row['f6_per_expert_ms']:>14.3f}{row['f6_layer_ms']:>13.1f}")
    for cname, c in checks.items():
        print(f"  {cname}: {'PASS' if c['ok'] else 'FAIL'}  {c}")
    return ok, {"checks": checks, "predicted_table": table,
                "note": "all figures estimator-predicted, not measured"}


# --------------------------------------------------------------------------- #
# GPU measurement side                                                          #
# --------------------------------------------------------------------------- #
def make_swiglu(inter, batched=False):
    def swiglu_ffn(x, W):
        gu = torch.bmm(x, W) if batched else x @ W
        g = gu[..., :inter]
        u = gu[..., inter:]
        return F.silu(g) * u
    return swiglu_ffn


# Dynamo's compiled-code cache lives ON the function's code object, and inductor config
# changes (mode / backends / is_big_gpu patch) are NOT part of its cache key -- compiling
# the same code object under a different variant could silently reuse the previous
# variant's artifact. Each compile variant therefore gets a textually distinct def.
def make_swiglu_nocg(inter, batched=False):
    def swiglu_ffn_nocg(x, W):
        gu = torch.bmm(x, W) if batched else x @ W
        g = gu[..., :inter]
        u = gu[..., inter:]
        return F.silu(g) * u
    return swiglu_ffn_nocg


def make_swiglu_forced(inter, batched=False):
    def swiglu_ffn_forced(x, W):
        gu = torch.bmm(x, W) if batched else x @ W
        g = gu[..., :inter]
        u = gu[..., inter:]
        return F.silu(g) * u
    return swiglu_ffn_forced


def residual_fn(x, Wo, res):
    return x @ Wo + res


def residual_fn_nocg(x, Wo, res):
    return x @ Wo + res


def residual_fn_forced(x, Wo, res):
    return x @ Wo + res


def compile_and_time(fn, args_tuple, iters, warmup, mode, forced=False):
    """Compile fn under `mode` (optionally with forced Triton GEMM templates), finish
    autotune with a few warm calls INSIDE the forced context (lowering happens on first
    call), then med_time + profile + numerics evidence. Returns a dict; never raises."""
    out = {"ms": None, "kernels": [], "fused": False, "evidence": {},
           "numerics_ok": None, "max_abs": None, "error": None}
    try:
        ctx = force_triton_templates() if forced else None
        if ctx:
            ctx.__enter__()
        try:
            cfn = torch.compile(fn, mode=mode, dynamic=False)
            for _ in range(4):
                cfn(*args_tuple)
            torch.cuda.synchronize()
            out["ms"] = med_time(lambda: cfn(*args_tuple), iters, warmup) * 1e3
            out["kernels"] = profile_kernels(lambda: cfn(*args_tuple))
            out["_cfn_out"] = cfn(*args_tuple)
        finally:
            if ctx:
                ctx.__exit__()
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return out


def measure_swiglu(name, M, hidden, inter, iters, warmup, gpu, gpu_adj,
                   count=1, batched=False, do_triton=True):
    """Measure + estimate one SwiGLU-into-up_gate config (dense count=1, or grouped MoE)."""
    print(f"[swiglu] {name}: M={M} hidden={hidden} inter={inter} "
          f"count={count} batched={batched} (autotune may take minutes) ...", flush=True)
    if batched:
        x = bf(count, M, hidden)
        Wug = bf(count, hidden, 2 * inter)
        dims = {"experts": count, "tokens_per_expert": M, "hidden": hidden, "inter": inter}
    else:
        x = bf(M, hidden)
        Wug = bf(hidden, 2 * inter)
        dims = {"M": M, "hidden": hidden, "inter": inter}
    swiglu_ffn = make_swiglu(inter, batched=batched)

    triton_ms = None
    triton_info = {}
    act_markers = ("silu", "sigmoid", "elementwise")
    with ClockSampler() as cs:
        gemm_only_s = med_time(lambda: (torch.bmm(x, Wug) if batched else x @ Wug), iters, warmup)
        unfused_s = med_time(lambda: swiglu_ffn(x, Wug), iters, warmup)
        ref = swiglu_ffn(x, Wug)
        # fp32 ground truth. The bf16 EAGER reference itself deviates from it (gu is rounded
        # to bf16 before silu); a path is numerically OK if it deviates no more than eager
        # does (2x margin, 5e-2 floor). The hand kernel's fp32 accumulators are MORE accurate
        # than eager, which a naive allclose-vs-eager wrongly fails at large k.
        gu32 = torch.bmm(x.float(), Wug.float()) if batched else x.float() @ Wug.float()
        ref32 = F.silu(gu32[..., :inter]) * gu32[..., inter:]
        del gu32
        eager_rel_max = _rel_max(ref, ref32)
        rel_tol = max(2.0 * eager_rel_max, 5e-2)

        # --- compiled paths ---
        # def   : plain max-autotune (what a user gets; cudagraphs ON -> the static-input
        #         copy kernel runs inside every timed call, tracked separately)
        # nocg  : max-autotune-no-cudagraphs, unforced -- on this 24-SM part inductor keeps
        #         the vendor GEMM + ONE separate fused pointwise epilogue kernel = the best
        #         UNFUSED realization (matches the estimator's unfused model: GEMM + a
        #         single 3x-inter-traffic vector kernel)
        # forced: Triton GEMM template forced (is_big_gpu patch + TRITON-only backends),
        #         no-cudagraphs -- the true fused-epilogue GEMM the task asks about
        r_def = compile_and_time(make_swiglu(inter, batched), (x, Wug), iters, warmup,
                                 "max-autotune")
        r_nocg = compile_and_time(make_swiglu_nocg(inter, batched), (x, Wug), iters, warmup,
                                  "max-autotune-no-cudagraphs")
        r_forced = compile_and_time(make_swiglu_forced(inter, batched), (x, Wug), iters, warmup,
                                    "max-autotune-no-cudagraphs", forced=True)
        for r in (r_def, r_nocg, r_forced):
            o = r.pop("_cfn_out", None)
            if o is not None:
                r["numerics_ok"] = bool(torch.allclose(o, ref, rtol=2e-2, atol=1e-2))
                r["max_abs"] = _maxabs(o, ref)
            if r["kernels"]:
                r["fused"], r["evidence"] = judge_fused(r["kernels"], act_markers)

        # --- hand Triton GEMM+SwiGLU with a quick tile autotune (dense only) ---
        t_ok = False
        if do_triton and _HAVE_TRITON and not batched:
            try:
                cands = TRITON_CANDS if iters >= 10 else TRITON_CANDS[:2]
                tcfg = tune_triton_swiglu(x, Wug, inter, cands)
                if tcfg is None:
                    triton_info = {"error": "all tile configs failed"}
                else:
                    BM, BN, BK, w, s = tcfg
                    tout = triton_swiglu(x, Wug, inter, BM, BN, BK, w, s)
                    t_rel = _rel_max(tout, ref32)
                    t_ok = bool(t_rel <= rel_tol)
                    # time it regardless; the verdict metric only uses it when numerics_ok
                    triton_s = med_time(
                        lambda: triton_swiglu(x, Wug, inter, BM, BN, BK, w, s),
                        iters, warmup)
                    triton_ms = triton_s * 1e3
                    triton_info = {"numerics_ok": t_ok,
                                   "rel_max_vs_fp32": t_rel,
                                   "eager_rel_max_vs_fp32": eager_rel_max,
                                   "rel_tol": rel_tol,
                                   "max_abs_diff_vs_eager": _maxabs(tout, ref),
                                   "config": f"BM{BM} BN{BN} BK{BK} w{w} s{s} "
                                             f"(tuned over {len(cands)} candidates)"}
            except Exception as exc:
                triton_info = {"error": f"{type(exc).__name__}: {exc}"}
        elif do_triton and batched:
            triton_info = {"skipped": "hand triton kernel is dense-only (bmm not adapted)"}

        # drift probe: re-measure the bare GEMM at the END of the config -- the spread vs the
        # first measurement bounds DVFS/thermal drift across this config's timed region
        gemm_repeat_s = med_time(lambda: (torch.bmm(x, Wug) if batched else x @ Wug),
                                 iters, warmup)
    clocks = cs.summary()
    sleep_cooldown()

    gemm_only_ms = gemm_only_s * 1e3
    unfused_ms = unfused_s * 1e3
    compiled_ms = r_def["ms"]
    compiled_nocg_ms = r_nocg["ms"]
    forced_ms = r_forced["ms"]
    fused_paths = {"compiled_ms": compiled_ms, "compiled_forced_triton_ms": forced_ms}
    if triton_ms is not None:
        fused_paths["triton_ms"] = triton_ms
    cand = [v for v in (compiled_ms, forced_ms, triton_ms) if v is not None]
    best_fused_ms = min(cand) if cand else None
    measured_gain = (unfused_ms / best_fused_ms) if best_fused_ms else None
    # verified-fused-only gain, against the BEST unfused realization: hand triton is fused
    # by construction; the forced template counts only if its kernel evidence confirms the fold
    unf_cand = [v for v in (unfused_ms, compiled_nocg_ms) if v is not None]
    best_unfused_ms = min(unf_cand)
    vcand = ([triton_ms] if (triton_ms is not None and t_ok) else []) + \
            ([forced_ms] if (forced_ms is not None and r_forced["fused"]) else [])
    measured_gain_verified = (best_unfused_ms / min(vcand)) if vcand else None
    compiled_over_gemm = (compiled_ms / gemm_only_ms) if compiled_ms else None
    nocg_over_gemm = (compiled_nocg_ms / gemm_only_ms) if compiled_nocg_ms else None
    forced_over_gemm = (forced_ms / gemm_only_ms) if forced_ms else None
    triton_over_gemm = (triton_ms / gemm_only_ms) if triton_ms else None

    est_unf, est_fus = est_swiglu_ms(M, hidden, inter, count, gpu)
    row = {
        "name": name, "kind": "swiglu", "dims": dims,
        "gemm_only_ms": gemm_only_ms, "unfused_ms": unfused_ms,
        "gemm_only_repeat_ms": gemm_repeat_s * 1e3,
        "gemm_drift_ratio": gemm_repeat_s * 1e3 / gemm_only_ms,
        "eager_rel_max_vs_fp32": eager_rel_max,
        "unfused_compiled_nocg_ms": compiled_nocg_ms,
        "best_unfused_ms": best_unfused_ms,
        "fused_paths": fused_paths,
        "best_fused_ms": best_fused_ms, "measured_gain": measured_gain,
        "measured_gain_verified": measured_gain_verified,
        "est_unfused_ms": est_unf, "est_fused_ms": est_fus,
        "estimated_gain": est_unf / est_fus,
        "fused_verified": r_def["fused"],
        "fused_verified_forced": r_forced["fused"],
        "kernel_evidence": r_def["kernels"],
        "fusion_evidence": r_def["evidence"],
        "nocg_kernel_evidence": r_nocg["kernels"],
        "nocg_fusion_evidence": r_nocg["evidence"],
        "forced_kernel_evidence": r_forced["kernels"],
        "forced_fusion_evidence": r_forced["evidence"],
        "numerics_ok": r_def["numerics_ok"], "max_abs_diff": r_def["max_abs"],
        "nocg_numerics_ok": r_nocg["numerics_ok"], "nocg_max_abs_diff": r_nocg["max_abs"],
        "forced_numerics_ok": r_forced["numerics_ok"], "forced_max_abs_diff": r_forced["max_abs"],
        "compiled_over_gemm_ratio": compiled_over_gemm,
        "nocg_over_gemm_ratio": nocg_over_gemm,
        "forced_over_gemm_ratio": forced_over_gemm,
        "triton_over_gemm_ratio": triton_over_gemm,
        "compiled_cudagraph_input_copy_us": _cudagraph_copy_us(r_def["kernels"]),
        "triton_info": triton_info,
        "compile_error": r_def["error"],
        "nocg_error": r_nocg["error"],
        "forced_error": r_forced["error"],
        "clocks": clocks,
    }
    if gpu_adj is not None:
        a_unf, a_fus = est_swiglu_ms(M, hidden, inter, count, gpu_adj)
        row["est_unfused_ms_adj"] = a_unf
        row["est_fused_ms_adj"] = a_fus
        row["estimated_gain_adj"] = a_unf / a_fus
    print(f"    gemm_only={gemm_only_ms:.4f}  unfused={unfused_ms:.4f}  "
          f"nocg={compiled_nocg_ms}  compiled={compiled_ms}  forced={forced_ms}  "
          f"triton={triton_ms}  meas_gain={measured_gain}  "
          f"gain_verified={measured_gain_verified}  est_gain={est_unf/est_fus:.4f}  "
          f"fused(def/forced)={r_def['fused']}/{r_forced['fused']}", flush=True)
    return row


def measure_residual(name, M, n, k, iters, warmup, gpu, gpu_adj):
    """Measure + estimate one residual-into-mla_o config (F1-analog, dense)."""
    print(f"[residual] {name}: M={M} n={n} k={k} (autotune may take minutes) ...", flush=True)
    x = bf(M, k)
    Wo = bf(k, n)
    res = bf(M, n)

    act_markers = ("add", "elementwise")
    with ClockSampler() as cs:
        gemm_only_s = med_time(lambda: x @ Wo, iters, warmup)
        unfused_s = med_time(lambda: x @ Wo + res, iters, warmup)
        addmm_s = med_time(lambda: torch.addmm(res, x, Wo), iters, warmup)
        ref = x @ Wo + res
        addmm_out = torch.addmm(res, x, Wo)
        addmm_ok = bool(torch.allclose(addmm_out, ref, rtol=2e-2, atol=1e-2))
        addmm_max_abs = _maxabs(addmm_out, ref)
        addmm_kernels = profile_kernels(lambda: torch.addmm(res, x, Wo))
        addmm_fused, addmm_separate = vendor_fused_ok(addmm_kernels, act_markers)

        # def/nocg/forced compiled variants (see measure_swiglu). For the residual epilogue
        # the unforced compile may legitimately fuse VENDOR-side (inductor emits addmm /
        # cublasLt beta-accumulate) -- judged via vendor_fused_ok on the kernel evidence.
        r_def = compile_and_time(residual_fn, (x, Wo, res), iters, warmup, "max-autotune")
        r_nocg = compile_and_time(residual_fn_nocg, (x, Wo, res), iters, warmup,
                                  "max-autotune-no-cudagraphs")
        r_forced = compile_and_time(residual_fn_forced, (x, Wo, res), iters, warmup,
                                    "max-autotune-no-cudagraphs", forced=True)
        for r in (r_def, r_nocg, r_forced):
            o = r.pop("_cfn_out", None)
            if o is not None:
                r["numerics_ok"] = bool(torch.allclose(o, ref, rtol=2e-2, atol=1e-2))
                r["max_abs"] = _maxabs(o, ref)
            if r["kernels"]:
                r["fused"], r["evidence"] = judge_fused(r["kernels"], act_markers)
                vok, vsep = vendor_fused_ok(r["kernels"], act_markers)
                r["evidence"]["vendor_fused_ok"] = vok
                r["evidence"]["vendor_separate_kernels"] = vsep

        # drift probe (see measure_swiglu)
        gemm_repeat_s = med_time(lambda: x @ Wo, iters, warmup)
    clocks = cs.summary()
    sleep_cooldown()

    gemm_only_ms = gemm_only_s * 1e3
    unfused_ms = unfused_s * 1e3
    addmm_ms = addmm_s * 1e3
    compiled_ms = r_def["ms"]
    compiled_nocg_ms = r_nocg["ms"]
    forced_ms = r_forced["ms"]
    fused_paths = {"addmm_ms": addmm_ms, "compiled_ms": compiled_ms,
                   "compiled_nocg_ms": compiled_nocg_ms,
                   "compiled_forced_triton_ms": forced_ms}
    cand = [v for v in (addmm_ms, compiled_ms, compiled_nocg_ms, forced_ms) if v is not None]
    best_fused_ms = min(cand) if cand else None
    measured_gain = (unfused_ms / best_fused_ms) if best_fused_ms else None
    # verified-fused-only gain: addmm counts if its profile shows no separate add kernel;
    # compiled paths count if Triton-template-fused OR vendor-fused per evidence
    vcand = ([addmm_ms] if addmm_fused else [])
    for r, ms in ((r_def, compiled_ms), (r_nocg, compiled_nocg_ms), (r_forced, forced_ms)):
        if ms is not None and (r["fused"] or r.get("evidence", {}).get("vendor_fused_ok")):
            vcand.append(ms)
    measured_gain_verified = (unfused_ms / min(vcand)) if vcand else None
    compiled_over_gemm = (compiled_ms / gemm_only_ms) if compiled_ms else None
    nocg_over_gemm = (compiled_nocg_ms / gemm_only_ms) if compiled_nocg_ms else None
    forced_over_gemm = (forced_ms / gemm_only_ms) if forced_ms else None
    addmm_over_gemm = addmm_ms / gemm_only_ms

    est_unf, est_fus = est_residual_ms(M, n, k, gpu)
    row = {
        "name": name, "kind": "residual", "dims": {"M": M, "n": n, "k": k},
        "gemm_only_ms": gemm_only_ms, "unfused_ms": unfused_ms,
        "gemm_only_repeat_ms": gemm_repeat_s * 1e3,
        "gemm_drift_ratio": gemm_repeat_s * 1e3 / gemm_only_ms,
        "best_unfused_ms": unfused_ms,
        "fused_paths": fused_paths,
        "best_fused_ms": best_fused_ms, "measured_gain": measured_gain,
        "measured_gain_verified": measured_gain_verified,
        "est_unfused_ms": est_unf, "est_fused_ms": est_fus,
        "estimated_gain": est_unf / est_fus,
        "fused_verified": r_def["fused"],
        "fused_verified_forced": r_forced["fused"],
        "addmm_fused_verified": addmm_fused,
        "addmm_kernel_evidence": addmm_kernels,
        "addmm_separate_kernels": addmm_separate,
        "kernel_evidence": r_def["kernels"],
        "fusion_evidence": r_def["evidence"],
        "nocg_kernel_evidence": r_nocg["kernels"],
        "nocg_fusion_evidence": r_nocg["evidence"],
        "forced_kernel_evidence": r_forced["kernels"],
        "forced_fusion_evidence": r_forced["evidence"],
        "numerics_ok": r_def["numerics_ok"], "max_abs_diff": r_def["max_abs"],
        "nocg_numerics_ok": r_nocg["numerics_ok"], "nocg_max_abs_diff": r_nocg["max_abs"],
        "forced_numerics_ok": r_forced["numerics_ok"], "forced_max_abs_diff": r_forced["max_abs"],
        "addmm_numerics_ok": addmm_ok, "addmm_max_abs_diff": addmm_max_abs,
        "compiled_over_gemm_ratio": compiled_over_gemm,
        "nocg_over_gemm_ratio": nocg_over_gemm,
        "forced_over_gemm_ratio": forced_over_gemm,
        "addmm_over_gemm_ratio": addmm_over_gemm,
        "compiled_cudagraph_input_copy_us": _cudagraph_copy_us(r_def["kernels"]),
        "compile_error": r_def["error"],
        "nocg_error": r_nocg["error"],
        "forced_error": r_forced["error"],
        "clocks": clocks,
    }
    if gpu_adj is not None:
        a_unf, a_fus = est_residual_ms(M, n, k, gpu_adj)
        row["est_unfused_ms_adj"] = a_unf
        row["est_fused_ms_adj"] = a_fus
        row["estimated_gain_adj"] = a_unf / a_fus
    print(f"    gemm_only={gemm_only_ms:.4f}  unfused={unfused_ms:.4f}  addmm={addmm_ms:.4f}  "
          f"nocg={compiled_nocg_ms}  compiled={compiled_ms}  forced={forced_ms}  "
          f"meas_gain={measured_gain}  gain_verified={measured_gain_verified}  "
          f"est_gain={est_unf/est_fus:.4f}  addmm_fused={addmm_fused}  "
          f"forced_fused={r_forced['fused']}", flush=True)
    return row


# --------------------------------------------------------------------------- #
# T6.C -- router top-k epilogue (attempt-and-DROP)                              #
# --------------------------------------------------------------------------- #
TOPK_MARKERS = ("topk", "sort", "radix", "select", "bitonic")


# Distinct code objects per compile variant (dynamo caches on the code object -- see
# make_swiglu_nocg note above).
def make_router_topk_chain():
    def router_topk_chain(x, W):
        return torch.topk(x @ W, TOP_K, dim=-1)
    return router_topk_chain


def make_router_topk_chain_nocg():
    def router_topk_chain_nocg(x, W):
        return torch.topk(x @ W, TOP_K, dim=-1)
    return router_topk_chain_nocg


def make_router_topk_chain_forced():
    def router_topk_chain_forced(x, W):
        return torch.topk(x @ W, TOP_K, dim=-1)
    return router_topk_chain_forced


def _topk_numerics(vals, idx, logits32, ref_vals, ref_idx, rel_tol):
    """C.4 numerics: sorted top-8 VALUES vs torch.topk(logits.float(), 8) under the T6.0.4
    rel-vs-fp32 criterion, plus a tie-insensitive index-SET check (gather the fp32 logits at
    the kernel's indices; the selected SET is right iff those values match the true top-8 --
    bf16 ties may legally pick different equal-valued experts, so raw set equality is only
    reported as a fraction, not the pass criterion)."""
    v_sorted = torch.sort(vals.float(), dim=-1, descending=True).values
    rel = _rel_max(v_sorted, ref_vals)
    vals_ok = bool(rel <= rel_tol)
    gathered = torch.sort(logits32.gather(1, idx.long()), dim=-1, descending=True).values
    idx_set_ok = bool(torch.allclose(gathered, ref_vals, rtol=2e-2, atol=1e-2))
    exact_frac = (torch.sort(idx.long(), dim=-1).values
                  == torch.sort(ref_idx, dim=-1).values).all(dim=-1).float().mean().item()
    return {"vals_ok": vals_ok, "rel_max_vs_fp32": rel, "rel_tol": rel_tol,
            "idx_set_ok": idx_set_ok, "exact_index_set_match_frac": exact_frac,
            "ok": bool(vals_ok and idx_set_ok)}


def measure_router_topk(name, M, k, regime, iters, warmup, gpu, gpu_adj, smoke=False):
    """T6.C: logits = x@Wr [M,256] then top-8. Baseline = min(cuBLAS+torch.topk, cuBLAS+
    standalone Triton row-topk). compiled/nocg/forced run purely as RECORDED EVIDENCE that a
    separate topk kernel survives (fusion barrier -> fused_verified False by construction).
    The one viable fused route is the hand BN=256 full-row kernel (C.4)."""
    n = EXPERTS                      # 256 -- the full-row (BN=N) design requires it
    print(f"[router_topk] {name}: M={M} n={n} k={k} ({regime}) ...", flush=True)
    x = bf(M, k)
    Wr = bf(k, n)

    with ClockSampler() as cs:
        gemm_only_s = med_time(lambda: x @ Wr, iters, warmup)
        logits = x @ Wr
        topk_only_s = med_time(lambda: torch.topk(logits, TOP_K, dim=-1), iters, warmup)
        eager_s = med_time(lambda: torch.topk(x @ Wr, TOP_K, dim=-1), iters, warmup)
        logits32 = x.float() @ Wr.float()
        ref_vals, ref_idx = torch.topk(logits32, TOP_K, dim=-1)
        e_vals, e_idx = torch.topk(logits, TOP_K, dim=-1)
        eager_rel = _rel_max(torch.sort(e_vals.float(), dim=-1, descending=True).values, ref_vals)
        rel_tol = max(2.0 * eager_rel, 5e-2)
        eager_num = _topk_numerics(e_vals, e_idx, logits32, ref_vals, ref_idx, rel_tol)

        # -- standalone Triton row-topk baseline (reads logits once; C.2) --
        rowtopk = {}
        rowtopk_chain_ms = None
        if _HAVE_TRITON:
            cands = ROWTOPK_CANDS if not smoke else ROWTOPK_CANDS[:2]
            tuned = _quick_tune(lambda cfg: (lambda: triton_row_topk(logits, *cfg)), cands)
            if tuned is None:
                rowtopk = {"error": "all row-topk tile configs failed"}
            else:
                rcfg = tuned[1]
                rv, ri = triton_row_topk(logits, *rcfg)
                rnum = _topk_numerics(rv, ri, logits32, ref_vals, ref_idx, rel_tol)
                r_only_s = med_time(lambda: triton_row_topk(logits, *rcfg), iters, warmup)
                r_chain_s = med_time(lambda: triton_row_topk(x @ Wr, *rcfg), iters, warmup)
                rowtopk = {"config": str(rcfg), "numerics": rnum,
                           "rowtopk_only_ms": r_only_s * 1e3,
                           "gemm_plus_rowtopk_ms": r_chain_s * 1e3}
                if rnum["ok"]:
                    rowtopk_chain_ms = r_chain_s * 1e3
        else:
            rowtopk = {"skipped": "triton unavailable"}

        # -- stock compile paths: recorded evidence only (C.3: cannot fuse a selection) --
        r_def = compile_and_time(make_router_topk_chain(), (x, Wr), iters, warmup,
                                 "max-autotune")
        r_nocg = compile_and_time(make_router_topk_chain_nocg(), (x, Wr), iters, warmup,
                                  "max-autotune-no-cudagraphs")
        r_forced = compile_and_time(make_router_topk_chain_forced(), (x, Wr), iters, warmup,
                                    "max-autotune-no-cudagraphs", forced=True)
        for r in (r_def, r_nocg, r_forced):
            o = r.pop("_cfn_out", None)
            if o is not None:
                r["topk_numerics"] = _topk_numerics(o[0], o[1], logits32, ref_vals, ref_idx,
                                                    rel_tol)
                r["numerics_ok"] = r["topk_numerics"]["ok"]
            if r["kernels"]:
                surv = [kk["name"] for kk in r["kernels"]
                        if any(mm in kk["name"].lower() for mm in TOPK_MARKERS)]
                r["topk_kernel_survives"] = surv
                # "fused" would mean NO separate topk/sort/select kernel remains -- the
                # structural expectation (and the recorded proof) is that one DOES survive.
                r["fused"] = len(surv) == 0
                r["evidence"] = {"n_kernels": len(r["kernels"]),
                                 "surviving_topk_kernels": surv}

        # -- the one viable fused path: hand BN=256 full-row GEMM+topk (C.4) --
        triton_fused_ms = None
        t_ok = False
        trit = {}
        if _HAVE_TRITON:
            cands = ROUTER_TOPK_CANDS if not smoke else ROUTER_TOPK_CANDS[:2]
            tuned = _quick_tune(lambda cfg: (lambda: triton_router_topk(x, Wr, *cfg)), cands)
            if tuned is None:
                trit = {"error": "all BN=256 fused tile configs failed to compile/launch"}
            else:
                tcfg = tuned[1]
                tv, ti_ = triton_router_topk(x, Wr, *tcfg)
                tnum = _topk_numerics(tv, ti_, logits32, ref_vals, ref_idx, rel_tol)
                t_ok = tnum["ok"]
                triton_fused_ms = med_time(lambda: triton_router_topk(x, Wr, *tcfg),
                                           iters, warmup) * 1e3
                tker = profile_kernels(lambda: triton_router_topk(x, Wr, *tcfg))
                trit = {"config": f"BM{tcfg[0]} BK{tcfg[1]} w{tcfg[2]} s{tcfg[3]} "
                                  f"(tuned over {len(cands)} candidates)",
                        "numerics": tnum, "numerics_ok": t_ok,
                        "kernels": tker, "n_kernels": len(tker),
                        "fused_by_construction": True}
        else:
            trit = {"skipped": "triton unavailable"}

        del logits32, ref_vals, ref_idx, logits
        gemm_repeat_s = med_time(lambda: x @ Wr, iters, warmup)
    clocks = cs.summary()
    sleep_cooldown()

    gemm_only_ms = gemm_only_s * 1e3
    eager_ms = eager_s * 1e3
    unf_cand = [v for v in (eager_ms, rowtopk_chain_ms) if v is not None]
    best_unfused_ms = min(unf_cand)
    best_fused_ms = triton_fused_ms          # the only candidate that CAN fuse (C.3)
    measured_gain = (best_unfused_ms / best_fused_ms) if best_fused_ms else None
    measured_gain_verified = measured_gain if (best_fused_ms and t_ok) else None
    stock_fused_any = any(r.get("fused") for r in (r_def, r_nocg, r_forced))
    drift_ratio = gemm_repeat_s * 1e3 / gemm_only_ms
    drift_clean = bool(abs(drift_ratio - 1.0) <= 0.05)

    try:
        est_unf, est_fus = est_router_topk_ms(M, n, k, gpu)
    except Exception as exc:
        est_unf = est_fus = None
        print(f"    est_router_topk failed: {exc}", flush=True)
    row = {
        "name": name, "kind": "router_topk", "regime": regime,
        "dims": {"M": M, "n": n, "k": k},
        "gemm_only_ms": gemm_only_ms,
        "gemm_only_repeat_ms": gemm_repeat_s * 1e3,
        "gemm_drift_ratio": drift_ratio, "drift_clean": drift_clean,
        "torch_topk_only_ms": topk_only_s * 1e3,
        "eager_unfused_ms": eager_ms,
        "rowtopk_baseline": rowtopk,
        "best_unfused_ms": best_unfused_ms,
        "stock_paths_evidence_only": {
            "compiled_ms": r_def["ms"], "compiled_nocg_ms": r_nocg["ms"],
            "compiled_forced_triton_ms": r_forced["ms"],
            "note": "recorded evidence per C.3: a separate topk/sort kernel survives on "
                    "every stock path (fusion barrier) -> not fused candidates"},
        "fused_paths": {"triton_fused_ms": triton_fused_ms},
        "best_fused_ms": best_fused_ms,
        "measured_gain": measured_gain,
        "measured_gain_verified": measured_gain_verified,
        "est_unfused_ms": est_unf, "est_fused_ms": est_fus,
        "estimated_gain": (est_unf / est_fus) if (est_unf and est_fus) else None,
        "est_traffic_only_bound": True,
        "est_excluded_from_aggregates": True,
        "eager_rel_max_vs_fp32": eager_rel,
        "eager_topk_numerics": eager_num,
        "fused_verified": r_def.get("fused", False),
        "fused_verified_forced": r_forced.get("fused", False),
        "stock_fused_any": stock_fused_any,
        "triton_info": trit,
        "numerics_ok": (trit.get("numerics_ok") if isinstance(trit, dict) else None),
        "kernel_evidence": r_def["kernels"],
        "fusion_evidence": r_def.get("evidence", {}),
        "nocg_kernel_evidence": r_nocg["kernels"],
        "nocg_fusion_evidence": r_nocg.get("evidence", {}),
        "forced_kernel_evidence": r_forced["kernels"],
        "forced_fusion_evidence": r_forced.get("evidence", {}),
        "compiled_topk_numerics": r_def.get("topk_numerics"),
        "nocg_topk_numerics": r_nocg.get("topk_numerics"),
        "forced_topk_numerics": r_forced.get("topk_numerics"),
        "compiled_cudagraph_input_copy_us": _cudagraph_copy_us(r_def["kernels"]),
        "compile_error": r_def["error"], "nocg_error": r_nocg["error"],
        "forced_error": r_forced["error"],
        "drop_condition_row": {"stock_fused": stock_fused_any,
                               "custom_gain_le_1": (measured_gain is not None
                                                    and measured_gain <= 1.0)},
        "clocks": clocks,
    }
    if gpu_adj is not None:
        try:
            a_unf, a_fus = est_router_topk_ms(M, n, k, gpu_adj)
            row["est_unfused_ms_adj"] = a_unf
            row["est_fused_ms_adj"] = a_fus
            row["estimated_gain_adj"] = a_unf / a_fus
        except Exception:
            pass
    print(f"    gemm_only={gemm_only_ms:.4f}  eager={eager_ms:.4f}  "
          f"best_unf={best_unfused_ms:.4f}  triton_fused={triton_fused_ms}  "
          f"meas_gain={measured_gain}  gain_ver={measured_gain_verified}  "
          f"est_bound={row['estimated_gain']}  stock_fused_any={stock_fused_any}  "
          f"drift={drift_ratio:.4f}", flush=True)
    return row


def eval_topk_drop(results):
    """C.6 DROP/KEEP condition booleans over the drift-clean router_topk rows (the drop TEXT
    belongs to the writeup; the condition evaluation is computed + stored here)."""
    rows = [r for r in results if r.get("kind") == "router_topk" and "error" not in r]
    clean = [r for r in rows if r.get("drift_clean")]
    with_gain = [r for r in clean if r.get("measured_gain") is not None]
    cond_a = (all(not r.get("stock_fused_any") for r in clean) if clean else None)
    cond_b = (all(r["measured_gain"] <= 1.0 for r in with_gain) if with_gain else None)
    wins = [{"name": r["name"], "measured_gain": r["measured_gain"],
             "triton_config": r.get("triton_info", {}).get("config")}
            for r in with_gain if r["measured_gain"] > 1.0]
    met = (bool(cond_a) and bool(cond_b)) if (cond_a is not None and cond_b is not None) else None
    return {
        "n_rows": len(rows), "n_drift_clean": len(clean), "n_with_custom_gain": len(with_gain),
        "cond_a_no_stock_path_fused": cond_a,
        "cond_b_custom_gain_le_1_all_clean": cond_b,
        "drop_condition_met": met,
        "keep_condition_wins": wins,
        "note": "DROP iff (a) AND (b) per C.6; KEEP (report the rare win) if any drift-clean "
                "config shows measured_gain > 1.0 with the custom kernel",
    }


# --------------------------------------------------------------------------- #
# T6.D -- MoE FFN fusion levels L1 / L2 / L3(F6), single expert + grouped        #
# --------------------------------------------------------------------------- #
def attempt_f6(x, Wug, Wdn, out32, rel_tol, iters, warmup, smoke=False):
    """D.4 ATTEMPT-and-record of the F6 hand kernel. Never raises: compile/launch/fit
    failures or bad numerics land in infeasible_reason (drop rules i/ii -> estimator-only)."""
    res = {"ms": None, "config": None, "numerics_ok": None, "rel_max_vs_fp32": None,
           "kernels": [], "n_kernels": None, "fused_verified": False,
           "infeasible_reason": None}
    if not _HAVE_TRITON:
        res["infeasible_reason"] = "triton unavailable"
        return res
    try:
        cands = F6_CANDS if not smoke else F6_CANDS[:2]
        tuned = _quick_tune(lambda cfg: (lambda: triton_f6_ffn(x, Wug, Wdn, *cfg)), cands)
        if tuned is None:
            res["infeasible_reason"] = (
                "F6 single-kernel infeasible in Triton on 99 KiB SMEM (D.4 drop rules i/ii). "
                "Evidence: (a) 4-chunk formulation (CH=INTER/4=512): triton OutOfResources at "
                "every candidate -- Required 176-322 KiB vs 101376 B limit (phase-2 dot staging "
                "of [512,BN_D] Wdn tiles + the persistent activated block; paper budget 84 KiB "
                "is unreachable because Triton stages every tl.dot operand in SMEM with no "
                "cross-phase buffer-reuse control); (b) 8-chunk variant (CH=256, BK_UG=32, "
                "stages=1) COMPILES within SMEM but returns corrupt results (rel_max 57-252 vs "
                "fp32, config-dependent) while the identical phase-1 chunk in an isolated "
                "kernel is numerically correct (rel 0.011) -- a Triton codegen/SMEM-liveness "
                "failure with 8 persistent dot operands. A CUTLASS-class kernel with explicit "
                "SMEM management could still realize the 84 KiB budget -> estimator-only F6.")
            return res
        cfg = tuned[1]
        o = triton_f6_ffn(x, Wug, Wdn, *cfg)
        rel = _rel_max(o, out32)
        ok = bool(rel <= rel_tol)
        res["config"] = (f"BM16 BK_ug{cfg[0]} BN_d{cfg[1]} w{cfg[2]} s{cfg[3]} "
                         f"(BN_ug=BK_d=INTER/4 fixed by static-chunk formulation; "
                         f"tuned over {len(cands)} candidates)")
        res["rel_max_vs_fp32"] = rel
        res["numerics_ok"] = ok
        res["ms"] = med_time(lambda: triton_f6_ffn(x, Wug, Wdn, *cfg), iters, warmup) * 1e3
        ks = profile_kernels(lambda: triton_f6_ffn(x, Wug, Wdn, *cfg))
        res["kernels"] = ks
        res["n_kernels"] = len(ks)
        res["fused_verified"] = bool(ok and len(ks) == 1)  # ONE kernel by construction
        if not ok:
            res["infeasible_reason"] = (f"numerics failed (rel_max {rel:.4g} > tol "
                                        f"{rel_tol:.4g}) -> estimator-only F6 (D.4 drop rule ii)")
    except Exception as exc:
        res["infeasible_reason"] = (f"{type(exc).__name__}: {exc} -> estimator-only F6 "
                                    f"(D.4 drop rule i)")
    return res


def _sw_best_verified_fused(sw):
    """Best VERIFIED-fused up_gate+SwiGLU ms from a measure_swiglu row (hand triton is fused
    by construction iff numerics ok; forced template iff fused_verified_forced)."""
    fp = sw.get("fused_paths", {}) or {}
    ti = sw.get("triton_info", {}) or {}
    cand = []
    if fp.get("triton_ms") is not None and ti.get("numerics_ok"):
        cand.append(fp["triton_ms"])
    if fp.get("compiled_forced_triton_ms") is not None and sw.get("fused_verified_forced"):
        cand.append(fp["compiled_forced_triton_ms"])
    if cand:
        return min(cand), True
    return sw.get("best_fused_ms"), False


def measure_ffn_levels(name, tpe, regime, iters, warmup, gpu, gpu_adj, smoke=False):
    """T6.D single-expert FFN at GLM dims: L1 (3 kernels) vs L2 (SwiGLU-in-up_gate epilogue
    + vendor down, via measure_swiglu) vs L3 (F6 one-kernel hand Triton). Layer = 256x."""
    hidden, inter = HIDDEN, INTERMEDIATE
    tokens = tpe * (EXPERTS // TOP_K)     # tpe = tokens/32
    mt = tpe // 16
    print(f"[ffn] {name}: tpe={tpe} tokens={tokens} ({regime}) mt={mt} ...", flush=True)

    # L2 (and the up_gate+SwiGLU part of L1) via the existing T4 machinery, verbatim (D.3)
    sw = measure_swiglu(f"{name}_upswiglu_L2", tpe, hidden, inter, iters, warmup, gpu,
                        gpu_adj, count=1, batched=False)

    x = bf(tpe, hidden)
    Wug = bf(hidden, 2 * inter)
    Wdn = bf(inter, hidden)
    with ClockSampler() as cs:
        up_gemm_s = med_time(lambda: x @ Wug, iters, warmup)
        gu = x @ Wug
        swiglu_vec_s = med_time(lambda: F.silu(gu[..., :inter]) * gu[..., inter:],
                                iters, warmup)
        act = F.silu(gu[..., :inter]) * gu[..., inter:]
        down_s = med_time(lambda: act @ Wdn, iters, warmup)
        gu32 = x.float() @ Wug.float()
        out32 = (F.silu(gu32[..., :inter]) * gu32[..., inter:]) @ Wdn.float()
        del gu32
        eager_rel = _rel_max(act @ Wdn, out32)
        rel_tol = max(2.0 * eager_rel, 5e-2)
        f6 = attempt_f6(x, Wug, Wdn, out32, rel_tol, iters, warmup, smoke=smoke)
        del out32
        up_repeat_s = med_time(lambda: x @ Wug, iters, warmup)   # drift probe (bare up_gate)
    clocks = cs.summary()
    sleep_cooldown()

    up_gemm_ms = up_gemm_s * 1e3
    swiglu_vec_ms = swiglu_vec_s * 1e3
    down_ms = down_s * 1e3
    up_swiglu_best_fused_ms, l2_verified = _sw_best_verified_fused(sw)
    ffn_L1_ms = up_gemm_ms + swiglu_vec_ms + down_ms         # eager 3-kernel sum (D.2)
    ffn_L1_best_ms = (sw["best_unfused_ms"] + down_ms        # nocg-based best unfused (D.2)
                      if sw.get("best_unfused_ms") is not None else None)
    ffn_L2_ms = (up_swiglu_best_fused_ms + down_ms
                 if up_swiglu_best_fused_ms is not None else None)
    ffn_L3_ms = f6["ms"] if f6.get("numerics_ok") else None  # measured only if numerics OK
    base = ffn_L1_best_ms if ffn_L1_best_ms is not None else ffn_L1_ms
    r_L2 = (base / ffn_L2_ms) if ffn_L2_ms else None
    r_L3 = (base / ffn_L3_ms) if ffn_L3_ms else None
    r_up_swiglu = ((sw["best_unfused_ms"] / up_swiglu_best_fused_ms)
                   if (sw.get("best_unfused_ms") and up_swiglu_best_fused_ms) else None)

    est_L1, est_L2, est_L3, est_m0, est_mt = est_ffn_levels_ms(tpe, hidden, inter, gpu)
    row = {
        "name": name, "kind": "ffn_levels", "regime": regime,
        "tpe": tpe, "tokens": tokens, "mt": mt,
        "dims": {"tpe": tpe, "hidden": hidden, "inter": inter,
                 "up_gate": [tpe, 2 * inter, hidden], "down": [tpe, hidden, inter]},
        "up_gate_gemm_ms": up_gemm_ms, "swiglu_vec_ms": swiglu_vec_ms,
        "down_gemm_ms": down_ms,
        "ffn_L1_ms": ffn_L1_ms, "ffn_L1_best_ms": ffn_L1_best_ms,
        "up_swiglu_best_fused_ms": up_swiglu_best_fused_ms,
        "up_swiglu_verified": l2_verified,
        "ffn_L2_ms": ffn_L2_ms, "ffn_L3_ms": ffn_L3_ms,
        "r_L2": r_L2, "r_L3": r_L3, "r_up_swiglu": r_up_swiglu,
        "est_ffn_L1_ms": est_L1, "est_ffn_L2_ms": est_L2, "est_ffn_L3_ms": est_L3,
        "est_r_L2": (est_L1 / est_L2) if est_L2 else None,
        "est_r_L3": (est_L1 / est_L3) if est_L3 else None,
        "f6_ms": f6["ms"], "f6_config": f6["config"],
        "f6_fused_verified": f6["fused_verified"],
        "f6_numerics_ok": f6["numerics_ok"],
        "f6_rel_max_vs_fp32": f6["rel_max_vs_fp32"],
        "f6_infeasible_reason": f6["infeasible_reason"],
        "f6_m0": (16 if f6["ms"] is not None else None), "f6_mt": mt,
        "est_f6_m0": est_m0, "est_f6_mt": est_mt,
        "f6_kernel_evidence": f6["kernels"], "f6_n_kernels": f6["n_kernels"],
        "eager_rel_max_vs_fp32": eager_rel,
        "forced_over_gemm_ratio": sw.get("forced_over_gemm_ratio"),
        "gemm_drift_ratio": up_repeat_s * 1e3 / up_gemm_ms,
        "drift_clean": bool(abs(up_repeat_s * 1e3 / up_gemm_ms - 1.0) <= 0.05),
        "layer_L1_ms": ffn_L1_ms * EXPERTS,
        "layer_L2_ms": (ffn_L2_ms * EXPERTS) if ffn_L2_ms else None,
        "layer_L3_ms": (ffn_L3_ms * EXPERTS) if ffn_L3_ms else None,
        "l2_swiglu_row": sw,
        "clocks": clocks,
    }
    if gpu_adj is not None:
        a1, a2, a3, _, _ = est_ffn_levels_ms(tpe, hidden, inter, gpu_adj)
        row["est_ffn_L1_ms_adj"] = a1
        row["est_ffn_L2_ms_adj"] = a2
        row["est_ffn_L3_ms_adj"] = a3
        row["est_r_L2_adj"] = (a1 / a2) if a2 else None
        row["est_r_L3_adj"] = (a1 / a3) if a3 else None
    print(f"    L1={ffn_L1_ms:.4f} (best {ffn_L1_best_ms})  L2={ffn_L2_ms}  L3={ffn_L3_ms}  "
          f"r_L2={r_L2}  r_L3={r_L3}  est_r_L2={row['est_r_L2']}  est_r_L3={row['est_r_L3']}  "
          f"f6_ok={f6['numerics_ok']}  f6_reason={f6['infeasible_reason']}", flush=True)
    return row


def measure_ffn_grouped(name, tpe, iters, warmup, gpu, gpu_adj, single_row, smoke=False):
    """T6.D.7 grouped-bmm 8-expert cross-check: L1/L2 via measure_swiglu(count=8,batched) +
    grouped down bmm; F6 with grid (8*mt,) (weights indexed per expert -> realistic
    occupancy, identical mt-x traffic penalty). Compares grouped vs 8x single-expert."""
    E = 8
    inter, hidden = INTERMEDIATE, HIDDEN
    mt = tpe // 16
    print(f"[ffn_grouped] {name}: E={E} tpe={tpe} mt={mt} grid=({E * mt},) ...", flush=True)
    sw = measure_swiglu(f"{name}_upswiglu", tpe, hidden, inter, iters, warmup, gpu, gpu_adj,
                        count=E, batched=True, do_triton=False)
    xg = bf(E, tpe, hidden)
    Wugg = bf(E, hidden, 2 * inter)
    Wdng = bf(E, inter, hidden)
    with ClockSampler() as cs:
        ug_s = med_time(lambda: torch.bmm(xg, Wugg), iters, warmup)
        gug = torch.bmm(xg, Wugg)
        sw_vec_s = med_time(lambda: F.silu(gug[..., :inter]) * gug[..., inter:],
                            iters, warmup)
        actg = F.silu(gug[..., :inter]) * gug[..., inter:]
        down_s = med_time(lambda: torch.bmm(actg, Wdng), iters, warmup)
        out32 = torch.empty(E, tpe, hidden, device=DEV, dtype=torch.float32)
        for e in range(E):                       # per-expert fp32 ref bounds transients
            gu32 = xg[e].float() @ Wugg[e].float()
            out32[e] = (F.silu(gu32[:, :inter]) * gu32[:, inter:]) @ Wdng[e].float()
            del gu32
        eager_rel = _rel_max(torch.bmm(actg, Wdng), out32)
        rel_tol = max(2.0 * eager_rel, 5e-2)
        f6 = attempt_f6(xg, Wugg, Wdng, out32, rel_tol, iters, warmup, smoke=smoke)
        del out32
        ug_repeat_s = med_time(lambda: torch.bmm(xg, Wugg), iters, warmup)
    clocks = cs.summary()
    sleep_cooldown()

    ug_ms, sw_vec_ms, down_ms = ug_s * 1e3, sw_vec_s * 1e3, down_s * 1e3
    up_swiglu_best_fused_ms, l2_verified = _sw_best_verified_fused(sw)
    grouped_L1_ms = ug_ms + sw_vec_ms + down_ms
    grouped_L1_best_ms = (sw["best_unfused_ms"] + down_ms
                          if sw.get("best_unfused_ms") is not None else None)
    grouped_L2_ms = (up_swiglu_best_fused_ms + down_ms
                     if up_swiglu_best_fused_ms is not None else None)
    grouped_L3_ms = f6["ms"] if f6.get("numerics_ok") else None
    base = grouped_L1_best_ms if grouped_L1_best_ms is not None else grouped_L1_ms

    def _vs_single(g_ms, key):
        s_ms = (single_row or {}).get(key)
        return (g_ms / (E * s_ms)) if (g_ms and s_ms) else None

    row = {
        "name": name, "kind": "ffn_grouped", "tpe": tpe, "experts": E, "mt": mt,
        "grouped_up_gate_bmm_ms": ug_ms, "grouped_swiglu_vec_ms": sw_vec_ms,
        "grouped_down_bmm_ms": down_ms,
        "grouped_L1_ms": grouped_L1_ms, "grouped_L1_best_ms": grouped_L1_best_ms,
        "grouped_up_swiglu_best_fused_ms": up_swiglu_best_fused_ms,
        "grouped_up_swiglu_verified": l2_verified,
        "grouped_L2_ms": grouped_L2_ms, "grouped_L3_ms": grouped_L3_ms,
        "grouped_r_L2": (base / grouped_L2_ms) if grouped_L2_ms else None,
        "grouped_r_L3": (base / grouped_L3_ms) if grouped_L3_ms else None,
        "grouped_over_8x_single_L1": _vs_single(grouped_L1_ms, "ffn_L1_ms"),
        "grouped_over_8x_single_L2": _vs_single(grouped_L2_ms, "ffn_L2_ms"),
        "grouped_over_8x_single_L3": _vs_single(grouped_L3_ms, "ffn_L3_ms"),
        "single_row_name": (single_row or {}).get("name"),
        "f6_ms": f6["ms"], "f6_config": f6["config"], "f6_grid": E * mt,
        "f6_fused_verified": f6["fused_verified"], "f6_numerics_ok": f6["numerics_ok"],
        "f6_rel_max_vs_fp32": f6["rel_max_vs_fp32"],
        "f6_infeasible_reason": f6["infeasible_reason"],
        "f6_kernel_evidence": f6["kernels"], "f6_n_kernels": f6["n_kernels"],
        "eager_rel_max_vs_fp32": eager_rel,
        "gemm_drift_ratio": ug_repeat_s * 1e3 / ug_ms,
        "drift_clean": bool(abs(ug_repeat_s * 1e3 / ug_ms - 1.0) <= 0.05),
        "l2_swiglu_row": sw,
        "clocks": clocks,
    }
    print(f"    grouped: L1={grouped_L1_ms:.4f}  L2={grouped_L2_ms}  L3={grouped_L3_ms}  "
          f"vs 8x single: L1={row['grouped_over_8x_single_L1']}  "
          f"L3={row['grouped_over_8x_single_L3']}  f6_ok={f6['numerics_ok']}", flush=True)
    return row


# --------------------------------------------------------------------------- #
# T6.E -- residual2 into the expert-merge reduction (no GEMM, memory-bound)      #
# --------------------------------------------------------------------------- #
def make_merge_r2f_fn():
    def merge_r2f(eo, g, r2):
        return (eo * g.unsqueeze(-1)).sum(dim=1) + r2
    return merge_r2f


def make_merge_r2f_nocg():
    def merge_r2f_nocg(eo, g, r2):
        return (eo * g.unsqueeze(-1)).sum(dim=1) + r2
    return merge_r2f_nocg


def make_merge_only_c():
    def merge_only_c(eo, g):
        return (eo * g.unsqueeze(-1)).sum(dim=1)
    return merge_only_c


def merge_fused_ok(kernels):
    """E-adapted fusion judge (vendor_fused_ok analogue, NO GEMM-template requirement):
    fused iff exactly ONE non-copy kernel remains and it is a triton_red/triton_poi kernel
    (i.e. the reduction absorbed the residual2 read; no trailing standalone add). cudagraph
    static-input copies (multi_tensor_apply) and Memcpy are excluded from the count."""
    cands = [k for k in kernels
             if "multi_tensor_apply" not in k["name"].lower()
             and "memcpy" not in k["name"].lower()]
    lows = [k["name"].lower() for k in cands]
    # inductor names the fused reduction triton_red_* / triton_poi_* / triton_per_*
    # (persistent reduction -- observed: triton_per_fused_add_mul_sum_unsqueeze_0)
    RED = ("triton_red", "triton_poi", "triton_per")
    reds = [k["name"] for k, l in zip(cands, lows) if any(p in l for p in RED)]
    trailing = [k["name"] for k, l in zip(cands, lows)
                if not any(p in l for p in RED)
                and any(m in l for m in ("add", "residual", "elementwise"))]
    fused = len(cands) == 1 and len(reds) == 1
    return fused, {"n_kernels_noncopy": len(cands), "reduction_kernels": reds,
                   "trailing_add_kernels": trailing,
                   "other_kernels": [k["name"] for k in cands if k["name"] not in reds]}


def _merge_rel_chunked(out, eo, g, r2, chunk=8192):
    """rel-vs-fp32 for the merge WITHOUT materializing the full fp32 reference (T=49152's
    fp32 ref alone is 1.2 GiB; chunking keeps peak memory within the E.7 budget)."""
    worst = 0.0
    for i in range(0, out.shape[0], chunk):
        sl = slice(i, min(i + chunk, out.shape[0]))
        ref = (eo[sl].float() * g[sl].float().unsqueeze(-1)).sum(dim=1) + r2[sl].float()
        worst = max(worst, ((out[sl].float() - ref).abs() / (ref.abs() + 1.0)).max().item())
        del ref
    return worst


def measure_merge_r2f(name, T, regime, iters, warmup, gpu, gpu_adj, smoke=False):
    """T6.E: out[t] = residual2[t] + sum_e gate[t,e]*expert_out[t,e]; synthetic dense TOP_K=8
    merge (T6.0.2b). compiled/nocg expected to FUSE stock (the strongest verdict-A shape);
    forced is N/A (no GEMM to template); baddbmm attempted once as a curiosity."""
    H = HIDDEN
    KE = TOP_K
    big = T >= 32768         # chunked numerics + no cudagraph path (static-input copy pool
    #                          would DOUBLE the ~5 GiB inputs -> OOM; E.7 memory budget)
    print(f"[merge_r2f] {name}: T={T} H={H} top_k={KE} ({regime}) big={big} ...", flush=True)
    eo = bf(T, KE, H)        # stacked-contiguous [T,8,H] (coalesced last dim)
    g = bf(T, KE)
    r2 = bf(T, H)

    def merge_only():
        return (eo * g.unsqueeze(-1)).sum(dim=1)

    def eager_fn():
        return (eo * g.unsqueeze(-1)).sum(dim=1) + r2

    with ClockSampler() as cs:
        merge_only_s = med_time(merge_only, iters, warmup)
        unfused_s = med_time(eager_fn, iters, warmup)
        e_out = eager_fn()
        eager_rel = _merge_rel_chunked(e_out, eo, g, r2)
        rel_tol = max(2.0 * eager_rel, 5e-2)
        del e_out
        torch.cuda.empty_cache() if big else None

        # clean-unfused: compiled merge-only (ONE fused reduction = estimator's 9*T*H model)
        # + separate eager residual add = the fair 2-kernel reference (E.2)
        clean_ms = None
        clean_err = None
        clean_kernels = []
        try:
            cmerge = torch.compile(make_merge_only_c(), mode="max-autotune-no-cudagraphs",
                                   dynamic=False)
            for _ in range(4):
                cmerge(eo, g)
            torch.cuda.synchronize()
            clean_ms = med_time(lambda: cmerge(eo, g) + r2, iters, warmup) * 1e3
            clean_kernels = profile_kernels(lambda: cmerge(eo, g) + r2)
        except Exception as exc:
            clean_err = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

        # fused stock paths
        if big:
            r_def = {"ms": None, "kernels": [], "fused": False, "evidence": {},
                     "numerics_ok": None, "max_abs": None,
                     "error": "skipped at T>=32768: cudagraph static-input pool would "
                              "double the ~5 GiB inputs > 8 GiB (nocg is the path of record)"}
        else:
            r_def = compile_and_time(make_merge_r2f_fn(), (eo, g, r2), iters, warmup,
                                     "max-autotune")
        r_nocg = compile_and_time(make_merge_r2f_nocg(), (eo, g, r2), iters, warmup,
                                  "max-autotune-no-cudagraphs")
        for r in (r_def, r_nocg):
            o = r.pop("_cfn_out", None)
            if o is not None:
                rel = _merge_rel_chunked(o, eo, g, r2)
                r["rel_max_vs_fp32"] = rel
                r["numerics_ok"] = bool(rel <= rel_tol)
                del o
            if r["kernels"]:
                r["fused"], r["evidence"] = merge_fused_ok(r["kernels"])

        # hand Triton kernel (optional upper bound, E.4)
        triton_ms = None
        t_ok = False
        trit = {}
        if _HAVE_TRITON:
            cands = MERGE_CANDS if not smoke else MERGE_CANDS[:2]
            tuned = _quick_tune(lambda cfg: (lambda: triton_merge_r2f(eo, g, r2, *cfg)), cands)
            if tuned is None:
                trit = {"error": "all merge tile configs failed"}
            else:
                mcfg = tuned[1]
                to = triton_merge_r2f(eo, g, r2, *mcfg)
                t_rel = _merge_rel_chunked(to, eo, g, r2)
                del to
                t_ok = bool(t_rel <= rel_tol)
                triton_ms = med_time(lambda: triton_merge_r2f(eo, g, r2, *mcfg),
                                     iters, warmup) * 1e3
                tker = profile_kernels(lambda: triton_merge_r2f(eo, g, r2, *mcfg))
                trit = {"config": f"BM{mcfg[0]} BN{mcfg[1]} w{mcfg[2]}",
                        "numerics_ok": t_ok, "rel_max_vs_fp32": t_rel,
                        "n_kernels": len(tker), "kernels": tker}
        else:
            trit = {"skipped": "triton unavailable"}

        # baddbmm curiosity (E.3): degenerate m=1,k=8 batched GEMM -- attempt once
        badd_ms = None
        badd_note = None
        badd_fused_ok = None
        badd_num_ok = None
        try:
            def badd():
                return torch.baddbmm(r2.unsqueeze(1), g.unsqueeze(1), eo).squeeze(1)
            bo = badd()
            b_rel = _merge_rel_chunked(bo, eo, g, r2)
            del bo
            badd_num_ok = bool(b_rel <= rel_tol)
            badd_ms = med_time(badd, iters, warmup) * 1e3
            bker = profile_kernels(badd)
            badd_fused_ok, badd_sep = vendor_fused_ok(bker, ("add", "residual", "elementwise"))
        except Exception as exc:
            badd_note = f"dropped: {type(exc).__name__}: {exc}"

        # drift probe = bare merge reduction re-measured at config end (T6.0.4)
        merge_repeat_s = med_time(merge_only, iters, warmup)
    clocks = cs.summary()
    sleep_cooldown()

    merge_only_ms = merge_only_s * 1e3
    unfused_ms = unfused_s * 1e3
    unf_cand = [v for v in (unfused_ms, clean_ms) if v is not None]
    best_unfused_ms = min(unf_cand)
    compiled_ms, nocg_ms = r_def["ms"], r_nocg["ms"]

    fused_paths = {"compiled_ms": compiled_ms, "compiled_nocg_ms": nocg_ms}
    if triton_ms is not None:
        fused_paths["triton_ms"] = triton_ms
    red_ref = min([v for v in (compiled_ms, nocg_ms) if v is not None] or [None])
    if badd_ms is not None and red_ref is not None:
        if badd_ms <= 1.5 * red_ref:
            fused_paths["baddbmm_ms"] = badd_ms
        else:
            badd_note = (f"dropped from fused_paths: baddbmm {badd_ms:.4f} ms > 1.5x fused "
                         f"reduction {red_ref:.4f} ms (tensor cores idle on m=1,k=8)")
    cand = [v for v in fused_paths.values() if v is not None]
    best_fused_ms = min(cand) if cand else None
    measured_gain = (best_unfused_ms / best_fused_ms) if best_fused_ms else None
    vcand = []
    if compiled_ms is not None and r_def.get("fused") and r_def.get("numerics_ok"):
        vcand.append(compiled_ms)
    if nocg_ms is not None and r_nocg.get("fused") and r_nocg.get("numerics_ok"):
        vcand.append(nocg_ms)
    if triton_ms is not None and t_ok:
        vcand.append(triton_ms)
    if "baddbmm_ms" in fused_paths and badd_fused_ok and badd_num_ok:
        vcand.append(badd_ms)
    measured_gain_verified = (best_unfused_ms / min(vcand)) if vcand else None
    # stock-compile-only variant (compiled/nocg, no hand kernel, no baddbmm) -- the
    # "torch.compile captures it out of the box" headline metric for E
    vcand_stock = [v for v, r in ((compiled_ms, r_def), (nocg_ms, r_nocg))
                   if v is not None and r.get("fused") and r.get("numerics_ok")]
    measured_gain_verified_stock = (best_unfused_ms / min(vcand_stock)) if vcand_stock else None
    drift_ratio = merge_repeat_s * 1e3 / merge_only_ms
    est_unf, est_fus = est_merge_r2f_ms(T, H, gpu)

    row = {
        "measured_gain_verified_stock": measured_gain_verified_stock,
        "name": name, "kind": "merge_r2f", "regime": regime,
        "dims": {"tokens": T, "top_k": KE, "hidden": H},
        "merge_only_ms": merge_only_ms,
        "merge_only_repeat_ms": merge_repeat_s * 1e3,
        "drift_ratio": drift_ratio,
        "drift_clean": bool(abs(drift_ratio - 1.0) <= 0.05),
        "unfused_ms": unfused_ms, "clean_unfused_ms": clean_ms,
        "clean_unfused_error": clean_err,
        "clean_unfused_kernels": clean_kernels,
        "best_unfused_ms": best_unfused_ms,
        "fused_paths": fused_paths,
        "forced": "N/A: no GEMM template (op is a reduction)",
        "baddbmm_attempt_ms": badd_ms, "baddbmm_note": badd_note,
        "baddbmm_fused_ok": badd_fused_ok, "baddbmm_numerics_ok": badd_num_ok,
        "best_fused_ms": best_fused_ms,
        "measured_gain": measured_gain,
        "measured_gain_verified": measured_gain_verified,
        "est_unfused_ms": est_unf, "est_fused_ms": est_fus,
        "estimated_gain": est_unf / est_fus,
        "estimated_gain_profile_and_token_independent": True,
        "fused_verified": r_def.get("fused", False),
        "fused_verified_nocg": r_nocg.get("fused", False),
        "kernel_evidence": r_def["kernels"],
        "fusion_evidence": r_def.get("evidence", {}),
        "nocg_kernel_evidence": r_nocg["kernels"],
        "nocg_fusion_evidence": r_nocg.get("evidence", {}),
        "eager_rel_max_vs_fp32": eager_rel,
        "numerics_ok": r_def.get("numerics_ok"),
        "rel_max_vs_fp32": r_def.get("rel_max_vs_fp32"),
        "nocg_numerics_ok": r_nocg.get("numerics_ok"),
        "nocg_rel_max_vs_fp32": r_nocg.get("rel_max_vs_fp32"),
        "triton_info": trit,
        "compiled_cudagraph_input_copy_us": _cudagraph_copy_us(r_def["kernels"]),
        "compile_error": r_def["error"], "nocg_error": r_nocg["error"],
        "clocks": clocks,
    }
    if gpu_adj is not None:
        a_unf, a_fus = est_merge_r2f_ms(T, H, gpu_adj)
        row["est_unfused_ms_adj"] = a_unf
        row["est_fused_ms_adj"] = a_fus
        row["estimated_gain_adj"] = a_unf / a_fus
        row["est_stock_equals_adjusted"] = bool(
            abs(row["estimated_gain"] - row["estimated_gain_adj"]) < 1e-9)
    print(f"    merge_only={merge_only_ms:.4f}  unfused={unfused_ms:.4f}  clean={clean_ms}  "
          f"compiled={compiled_ms}  nocg={nocg_ms}  triton={triton_ms}  badd={badd_ms}  "
          f"gain={measured_gain}  gain_ver={measured_gain_verified}  "
          f"est=1.20  fused(nocg)={r_nocg.get('fused')}  drift={drift_ratio:.4f}", flush=True)
    del eo, g, r2
    torch.cuda.empty_cache()
    return row


def merge_dropped_131072_row():
    """E.7: the 131072 prefill point is INFEASIBLE at 8 GB -- recorded with the arithmetic
    (coverage comes from the token-independence assertion, not silence)."""
    return {
        "name": "merge_prefill_T131072", "kind": "merge_r2f", "regime": "prefill",
        "dims": {"tokens": 131072, "top_k": 8, "hidden": HIDDEN},
        "dropped": True,
        "infeasible_reason": (
            "expert_outs[131072,8,6144] bf16 alone = 8*131072*6144*2 B = 12.9 GB > 8 GB "
            "VRAM; per-token resident ~144 KB (expert_outs 96 KB + residual2 12 KB + out "
            "12 KB + unfused merged 12 KB + transient fp32 ref 24 KB) -> T <~ 40k practical "
            "ceiling; 49152 (~7.8 GB peak, fp32 ref + merged freed before big allocs) is "
            "the borderline largest measurable point. Coverage: the E.5 token-independence "
            "assertion (gain flat 512->49152 => the ratio has no T dependence => the "
            "un-measurable 131072 gain is the same)."),
    }


def eval_merge_token_independence(results):
    """E.5/E.7: verified-fused gain at T=512/8192/32768/49152 must be flat for the dropped
    131072 point to be defensibly inferable. Returns the JSON block (bool + spread)."""
    req = [512, 8192, 32768, 49152]
    pts = {}
    for r in results:
        if (r.get("kind") == "merge_r2f" and "error" not in r and not r.get("dropped")
                and r.get("measured_gain_verified") is not None):
            pts[r["dims"]["tokens"]] = r["measured_gain_verified"]
    have = {str(t): pts[t] for t in req if t in pts}
    blk = {"required_points": req, "gains_by_tokens": have}
    if len(have) == len(req):
        vals = list(have.values())
        spread = (max(vals) - min(vals)) / (sum(vals) / len(vals))
        blk["spread_frac"] = spread
        blk["flat_threshold"] = 0.05
        blk["merge_gain_token_independent"] = bool(spread <= 0.05)
    else:
        blk["merge_gain_token_independent"] = None
        blk["note"] = (f"insufficient measured points ({sorted(pts)} of {req}) -- "
                       "cannot claim 131072 coverage")
    return blk


# --------------------------------------------------------------------------- #
# --t2-json adjusted estimator profile                                          #
# --------------------------------------------------------------------------- #
def _find_numeric(obj, patterns):
    """PRIORITY-ordered recursive key search: an earlier pattern beats a later one no
    matter where its key sits in the JSON (dict-walk order only breaks ties within one
    pattern). A walk-order-first search would return e.g. peak_gemm.shapes[0].tflops
    (the FIRST square GEMM, not the peak) from rtx4060_peak.py output."""
    pats = [p.lower() for p in patterns]
    per_pattern = [[] for _ in pats]

    def walk(o):
        if isinstance(o, dict):
            for kk, vv in o.items():
                if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                    kl = str(kk).lower()
                    for i, p in enumerate(pats):
                        if p in kl:
                            per_pattern[i].append(float(vv))
                            break
                walk(vv)
        elif isinstance(o, list):
            for it in o:
                walk(it)

    walk(obj)
    for hits in per_pattern:
        if hits:
            return hits[0]
    return None


def build_adjusted_profile(t2_path, gpu):
    """Adjusted GpuModel from T2's measured peaks.

    Preferred: rtx4060_peak.py writes an `adjusted_profile` block with the exact
    dataclasses.replace overrides (clock_hz, bw_bytes_per_s) it documents for
    downstream consumption -- use it verbatim. Fallback for unknown T2 schemas:
    priority-ordered recursive key search (T2's real key names first: T2 reports
    bandwidth as *_gbs, and its peak lives in measured_peak_tflops, NOT the
    per-shape tflops entries).
    """
    try:
        with open(t2_path) as f:
            data = json.load(f)
    except Exception as exc:
        return None, {"error": f"could not read {t2_path}: {exc}"}

    blk = data.get("adjusted_profile") if isinstance(data, dict) else None
    if (isinstance(blk, dict)
            and isinstance(blk.get("clock_hz"), (int, float))
            and isinstance(blk.get("bw_bytes_per_s"), (int, float))):
        kwargs = {"clock_hz": float(blk["clock_hz"]),
                  "bw_bytes_per_s": float(blk["bw_bytes_per_s"])}
        adj = dataclasses.replace(gpu, name=gpu.name + " +T2adj", **kwargs)
        return adj, {"applied": kwargs, "source": "t2 adjusted_profile block",
                     "check_peak_tflops": adj.peak_tensor_flops / 1e12,
                     "check_bw_gbs": adj.bw_bytes_per_s / 1e9}

    tflops = _find_numeric(data, ("measured_peak_tflops", "peak_bf16_tflops", "peak_tflops",
                                  "bf16_tflops", "measured_tflops", "tflops"))
    gbps = _find_numeric(data, ("measured_bw_gbs", "measured_bw_gbps", "hbm_gbps", "bw_gbps",
                                "bandwidth_gbps", "measured_gbps", "gbps", "gbs"))
    if tflops is None and gbps is None:
        return None, {"note": "no recognizable tflops/bandwidth keys in t2 json; adjusted profile skipped"}
    kwargs = {}
    if gbps is not None:
        kwargs["bw_bytes_per_s"] = gbps * 1e9
    if tflops is not None:
        # peak_tensor_flops is derived: tensor_cores * flops_per_core_per_clock * clock_hz
        kwargs["tensor_flops_per_core_per_clock"] = tflops * 1e12 / (gpu.tensor_cores * gpu.clock_hz)
    adj = dataclasses.replace(gpu, name=gpu.name + " +T2adj", **kwargs)
    return adj, {"applied": {k: kwargs[k] for k in kwargs}, "src_tflops": tflops,
                 "src_gbps": gbps, "source": "pattern search fallback"}


# --------------------------------------------------------------------------- #
# Config sweeps                                                                  #
# --------------------------------------------------------------------------- #
def swiglu_configs(smoke):
    if smoke:
        return [("swiglu_smoke", 512, 512, 512, 1, False)]
    cfgs = []
    for M in (2048, 8192):
        for hidden in (1024, 2048, 4096):
            cfgs.append((f"swiglu_M{M}_h{hidden}", M, hidden, hidden, 1, False))
    return cfgs


def residual_configs(smoke):
    if smoke:
        return [("residual_smoke", 512, 512, 512)]      # (name, M, n, k) tiny
    # task ordering: x[M,6144]@Wo[6144,16384]+res[M,16384]  (n=16384, k=6144)
    # GLM/C500 ordering: x[M,16384]@Wo[16384,6144]+res[M,6144]  (n=6144, k=16384) --
    # the actual mla_o+residual layer the C500 measured; both fit 8 GiB, measure both.
    return ([(f"residual_M{M}", M, 16384, 6144) for M in (2048, 8192)]
            + [(f"residual_glm_M{M}", M, 6144, 16384) for M in (2048, 8192)])


def moe_configs(smoke):
    # grouped-bmm MoE: (name, tokens_per_expert, hidden, inter, experts)
    if smoke:
        return [("moe_smoke", 64, 512, 512, 4)]
    cfgs = []
    for E in (8, 32):
        cfgs.append((f"moe_E{E}", 128, 2048, 2048, E))
    return cfgs


def router_topk_configs(smoke):
    # (name, M=tokens, k, regime); router is DENSE -> M = tokens directly; n=EXPERTS fixed.
    # tokens=8192 kept in BOTH regimes (T6.0.3 crossover anchor).
    if smoke:
        return [("topk_smoke_M256", 256, 512, "decode")]
    return ([(f"topk_decode_M{M}", M, HIDDEN, "decode")
             for M in (512, 1024, 2048, 4096, 8192, 16384)]
            + [(f"topk_prefill_M{M}", M, HIDDEN, "prefill")
               for M in (8192, 32768, 131072)])


def ffn_configs(smoke):
    # (name, tpe, regime); ALWAYS GLM dims (hidden=6144, inter=2048) so the estimator side
    # (estimate_ffn_fused_m reads the GLM globals) matches. tpe=256 (tokens=8192) in both.
    # Smoke: tpe=16 at full GLM dims (75 MB weights) exercises every path incl. F6.
    if smoke:
        return [("ffn_smoke_tpe16", 16, "decode")]
    return ([(f"ffn_decode_tpe{t}", t, "decode") for t in (16, 32, 64, 128, 256, 512)]
            + [(f"ffn_prefill_tpe{t}", t, "prefill") for t in (256, 1024, 4096)])


def ffn_grouped_configs(smoke):
    # D.7: 8-expert grouped cross-check at one decode + one prefill tpe.
    if smoke:
        return [("ffn_grouped_smoke_tpe16", 16)]
    return [("ffn_grouped_tpe64", 64), ("ffn_grouped_tpe512", 512)]


def merge_configs(smoke):
    # (name, T=tokens, regime); 131072 dropped separately (merge_dropped_131072_row).
    if smoke:
        return [("merge_smoke_T512", 512, "decode")]
    return ([(f"merge_decode_T{T}", T, "decode")
             for T in (512, 1024, 2048, 4096, 8192, 16384)]
            + [(f"merge_prefill_T{T}", T, "prefill") for T in (32768, 49152)])


# T6 conventions, merged into the JSON's conventions block via setdefault (never
# overwriting existing T4 keys on merge-append).
T6_CONVENTIONS = {
    "t6_merge_append": "with --topk/--ffn/--merge, an existing --out is loaded and new rows "
                       "are APPENDED to configs (existing rows + annotations_post_hoc "
                       "preserved); default T4 invocation still overwrites as before",
    "router_topk": "T6.C attempt-and-DROP: best_unfused = min(cuBLAS+torch.topk, cuBLAS+"
                   "standalone Triton row-topk); compiled/nocg/forced are recorded evidence "
                   "that a separate topk kernel survives (fusion barrier); the only fused "
                   "candidate is the hand BN=256 full-row kernel; estimated_gain is a "
                   "STRUCTURE-BLIND traffic-only bound (est_traffic_only_bound=true), "
                   "excluded from every aggregate; drop booleans in router_topk_drop_evaluation",
    "ffn_levels": "T6.D single-expert GLM FFN: ffn_L1_ms = eager up_gate+SwiGLU+down 3-kernel "
                  "sum; ffn_L1_best_ms = best_unfused(up+SwiGLU incl. nocg) + vendor down "
                  "(ratios use this per D.2); ffn_L2_ms = best VERIFIED-fused up_gate+SwiGLU "
                  "(hand triton / forced template, from the embedded measure_swiglu row) + "
                  "vendor down; ffn_L3_ms = one-kernel F6 hand triton (grid (mt,), BM=16 "
                  "hard cap) iff numerics pass, else estimator-only with f6_infeasible_reason; "
                  "layer_L*_ms = 256 x per-expert (T6.0.2: balanced routing, merge excluded)",
    "ffn_grouped": "T6.D.7 8-expert grouped-bmm cross-check; F6 grid=(8*mt,) with weights "
                   "indexed per expert (restores occupancy; identical mt-x traffic penalty); "
                   "grouped_over_8x_single_* compares to 8x the same-tpe single-expert row",
    "merge_r2f": "T6.E synthetic dense TOP_K=8 merge + residual2 (no GEMM): best_unfused = "
                 "min(eager two-op, compiled merge-only + separate eager add); fused = "
                 "compiled/nocg (stock, judged by merge_fused_ok: ONE triton_red/poi kernel, "
                 "no trailing add), optional hand triton, baddbmm curiosity (kept iff <=1.5x "
                 "fused reduction); forced N/A (no GEMM template); estimated_gain = 12/10 = "
                 "1.20 exactly, profile- and token-independent (stock == T2-adjusted); drift "
                 "probe = bare merge reduction; 131072 dropped with arithmetic, covered by "
                 "the token-independence assertion (merge_token_independence block)",
    "drift_clean": "|drift_ratio - 1| <= 0.05 (T6.0.4); drift-tainted rows stay in the JSON "
                   "but are excluded from aggregates",
}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny dims, iters=3/warmup=3, whole path end-to-end")
    ap.add_argument("--moe", action="store_true", help="add grouped-bmm MoE swiglu configs")
    ap.add_argument("--topk", action="store_true",
                    help="T6.C router top-k epilogue rows ONLY (skips the T4 groups; "
                         "merge-appends to an existing --out)")
    ap.add_argument("--ffn", action="store_true",
                    help="T6.D FFN L1/L2/L3(F6) rows ONLY (D.5 estimator acceptance runs "
                         "first and HALTS on failure; merge-appends)")
    ap.add_argument("--merge", action="store_true",
                    help="T6.E expert-merge + residual2 rows ONLY (merge-appends)")
    ap.add_argument("--t2-json", default=None,
                    help="T2 measured-peaks JSON; also report estimator gains under an adjusted profile")
    args = ap.parse_args()
    t6_mode = args.topk or args.ffn or args.merge

    # Fail fast on an unwritable --out BEFORE spending GPU time on compiles.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    iters, warmup = (3, 3) if args.smoke else (30, 15)

    gpu = GPUS["rtx4060-measured"]
    gpu_adj, adj_info = (None, None)
    if args.t2_json:
        gpu_adj, adj_info = build_adjusted_profile(args.t2_json, gpu)

    props = torch.cuda.get_device_properties(0)
    env = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "triton_version": (triton.__version__ if _HAVE_TRITON else None),
        "device_name": torch.cuda.get_device_name(0),
        "sm_count": props.multi_processor_count,
        "total_memory_gib": round(props.total_memory / 1024**3, 4),
        "smem_per_block_optin_bytes": getattr(props, "shared_memory_per_block_optin", None),
        "estimator_profile": gpu.name,
        "estimator_peak_tflops": gpu.peak_tensor_flops / 1e12,
        "estimator_bw_gbps": gpu.bw_bytes_per_s / 1e9,
        "clocks_locked": False,
        "clocks_note": "WSL2 / no root: clocks NOT locked -> ClockSampler recorded per config",
        "smoke": args.smoke,
        "adjusted_profile": adj_info,
    }
    conventions = {
        "time_unit": "milliseconds for every field ending in _ms",
        "med_time_returns": "SECONDS (rtx4060_common.med_time); converted to ms here (*1e3)",
        "estimator_time_unit": "SECONDS (KTime.time_s); converted to ms here (*1e3)",
        "gain_definition": "gain = unfused_time / fused_time  (>1.0 => fusion FASTER), matches estimator '1.020x'",
        "measured_gain": "unfused_ms / best_fused_ms  (best over compiled/addmm/triton paths)",
        "estimated_gain": "est_unfused_ms / est_fused_ms  (fusion_time_estimator, rtx4060-measured profile)",
        "dtype": "bfloat16 for all measured tensors; fp32 accumulate",
        "hand_triton_numerics": "judged vs an fp32 reference: ok iff rel_max_vs_fp32 <= "
                          "max(2 x eager path's own rel_max_vs_fp32, 5e-2). The hand kernel keeps "
                          "gate/up in fp32 accumulators (MORE accurate than eager, which rounds gu "
                          "to bf16 pre-silu), so allclose-vs-eager is the wrong criterion at large k",
        "gemm_drift_ratio": "bare-GEMM time re-measured at the END of the config / at the START; "
                          "bounds DVFS+thermal drift across the config's timed region (~1.0 = clean)",
        "fused_verified": "True iff a Triton GEMM template folds the epilogue in AND no separate "
                          "silu/mul/add elementwise kernel remains (raw kernel_evidence stored for re-judge)",
        "compiled_paths": "compiled_ms = plain max-autotune (cudagraphs ON; is_big_gpu()<68 SMs so "
                          "inductor keeps the vendor GEMM + separate epilogue kernel on this GPU); "
                          "unfused_compiled_nocg_ms / compiled_nocg_ms = max-autotune-no-cudagraphs "
                          "unforced (for swiglu this is the BEST UNFUSED realization: vendor GEMM + one "
                          "fused pointwise kernel, exactly the estimator's unfused model); "
                          "compiled_forced_triton_ms = is_big_gpu patched + TRITON-only autotune "
                          "backends + no-cudagraphs -> a genuine Triton GEMM template with the epilogue "
                          "folded in (the object the task asks about); fused_verified_forced judges it",
        "measured_gain_verified": "best_unfused_ms / best VERIFIED-fused path (hand triton by "
                          "construction; forced template iff fused_verified_forced; addmm iff "
                          "addmm_fused_verified; compiled paths iff template- or vendor-fused per "
                          "evidence). The A-vs-B verdict keys on THIS, not measured_gain",
        "addmm_fused_verified": "residual only: torch.addmm profile shows no separate add/elementwise "
                          "kernel (Memcpy DtoD tolerated: beta-accumulate copy)",
        "compiled_cudagraph_input_copy_us": "GPU time of multi_tensor_apply static-input-copy kernels "
                          "inside ONE compiled call (cudagraph deployment overhead, not fusion work); "
                          "subtract from compiled_ms for a kernels-only fused-vs-unfused comparison",
        "timing": f"median of {iters} cuda-event samples, {warmup} warmup, per config",
    }

    print(f"=== T4 fusion realizability  (smoke={args.smoke}, moe={args.moe}, "
          f"topk={args.topk}, ffn={args.ffn}, merge={args.merge}) ===", flush=True)
    print(f"device={env['device_name']} torch={env['torch_version']} triton={env['triton_version']}",
          flush=True)

    # T6 merge-append: with any new flag, an existing --out is LOADED and appended to
    # (existing configs rows + annotations_post_hoc preserved verbatim); the default T4
    # invocation keeps its original fresh-overwrite behavior.
    if t6_mode and os.path.exists(args.out):
        try:
            with open(args.out) as f:
                out = json.load(f)
        except Exception as exc:
            print(f"WARNING: could not load existing {args.out} ({exc}); starting fresh",
                  flush=True)
            out = {}
        results = out.setdefault("configs", [])
        conv = out.setdefault("conventions", {})
        for kk, vv in conventions.items():
            conv.setdefault(kk, vv)
        out.setdefault("env", env)
        out["env_t6"] = env                # this run's env, without clobbering the T4 env
        print(f"[merge-append] loaded {args.out}: {len(results)} existing config rows kept",
              flush=True)
    else:
        results = []
        out = {"conventions": conventions, "env": env, "configs": results}
    for kk, vv in T6_CONVENTIONS.items():
        out["conventions"].setdefault(kk, vv)

    def run_config(kind, name, thunk):
        """Isolate one config: an exception records an error row instead of killing the
        sweep, and the JSON is re-saved after EVERY config (partial-result safety)."""
        try:
            row = thunk()
        except Exception as exc:
            traceback.print_exc()
            row = {"name": name, "kind": kind, "error": f"{type(exc).__name__}: {exc}"}
        results.append(row)
        save_json(args.out, out)

    if not t6_mode:
        for name, M, hidden, inter, count, batched in swiglu_configs(args.smoke):
            run_config("swiglu", name,
                       lambda: measure_swiglu(name, M, hidden, inter, iters, warmup, gpu, gpu_adj,
                                              count=count, batched=batched))
        for name, M, n, k in residual_configs(args.smoke):
            run_config("residual", name,
                       lambda: measure_residual(name, M, n, k, iters, warmup, gpu, gpu_adj))
        if args.moe:
            for name, tpe, hidden, inter, E in moe_configs(args.smoke):
                run_config("swiglu", name,
                           lambda: measure_swiglu(name, tpe, hidden, inter, iters, warmup, gpu,
                                                  gpu_adj, count=E, batched=True, do_triton=False))

    if args.topk:
        for name, M, kdim, regime in router_topk_configs(args.smoke):
            run_config("router_topk", name,
                       lambda: measure_router_topk(name, M, kdim, regime, iters, warmup,
                                                   gpu, gpu_adj, smoke=args.smoke))
        out["router_topk_drop_evaluation"] = eval_topk_drop(results)
        save_json(args.out, out)

    if args.ffn:
        # D.5 HARD PREREQUISITE: the tpe-parametrized F6 estimator must pass acceptance
        # BEFORE any F6 est is trusted / the sweep runs.
        acc_ok, acc_report = f6_estimator_acceptance(gpu)
        out["f6_estimator_acceptance"] = acc_report
        save_json(args.out, out)
        if not acc_ok:
            print("\n[HALT] T6.D.5 estimator acceptance FAILED: estimate_ffn_fused_m does "
                  "not reproduce the known F6=0.259x at m=64 / the mt weight-reread scaling "
                  "/ the m0=16 SMEM cap. Fix fusion_time_estimator.estimate_ffn_fused_m "
                  "before running the D sweep -- every F6 est would be untrustworthy.",
                  flush=True)
            sys.exit(2)
        single_by_tpe = {}
        for name, tpe, regime in ffn_configs(args.smoke):
            run_config("ffn_levels", name,
                       lambda: measure_ffn_levels(name, tpe, regime, iters, warmup,
                                                  gpu, gpu_adj, smoke=args.smoke))
            last = results[-1]
            if last.get("kind") == "ffn_levels" and "error" not in last:
                single_by_tpe.setdefault(tpe, last)
        for name, tpe in ffn_grouped_configs(args.smoke):
            run_config("ffn_grouped", name,
                       lambda: measure_ffn_grouped(name, tpe, iters, warmup, gpu, gpu_adj,
                                                   single_by_tpe.get(tpe), smoke=args.smoke))

    if args.merge:
        for name, T, regime in merge_configs(args.smoke):
            run_config("merge_r2f", name,
                       lambda: measure_merge_r2f(name, T, regime, iters, warmup,
                                                 gpu, gpu_adj, smoke=args.smoke))
        if not args.smoke and not any(r.get("name") == "merge_prefill_T131072"
                                      for r in results):
            results.append(merge_dropped_131072_row())
        ti_blk = eval_merge_token_independence(results)
        out["merge_token_independence"] = ti_blk
        for r in results:
            if r.get("kind") == "merge_r2f" and "error" not in r and not r.get("dropped"):
                r["merge_gain_token_independent"] = ti_blk["merge_gain_token_independent"]
        save_json(args.out, out)

    # -------- human-readable summary --------
    T4_KINDS = ("swiglu", "residual")
    t4_rows = [r for r in results if r.get("kind") in T4_KINDS]
    ok_rows = [r for r in t4_rows if "error" not in r]
    err_rows = [r for r in t4_rows if "error" in r]
    print("\n" + "=" * 130)
    print(f"{'config':<20}{'kind':<10}{'unfused':>9}{'best_fus':>10}{'meas_gain':>10}"
          f"{'gain_ver':>9}{'est_gain':>9}{'forced?':>8}{'num_ok':>7}{'forc/gemm':>10}")
    print("-" * 130)
    for r in ok_rows:
        bf_ms = r["best_fused_ms"]
        mg = r["measured_gain"]
        mgv = r.get("measured_gain_verified")
        fog = r.get("forced_over_gemm_ratio")
        print(f"{r['name']:<20}{r['kind']:<10}{r['unfused_ms']:>9.4f}"
              f"{(bf_ms if bf_ms is not None else float('nan')):>10.4f}"
              f"{(mg if mg is not None else float('nan')):>10.4f}"
              f"{(mgv if mgv is not None else float('nan')):>9.4f}"
              f"{r['estimated_gain']:>9.4f}{str(r.get('fused_verified_forced')):>8}"
              f"{str(r['numerics_ok']):>7}"
              f"{(fog if fog is not None else float('nan')):>10.4f}")
    for r in err_rows:
        print(f"{r['name']:<20}{r['kind']:<10}  ERROR: {r['error']}")
    print("-" * 130)
    meas_gains = [r["measured_gain"] for r in ok_rows if r["measured_gain"]]
    ver_gains = [r["measured_gain_verified"] for r in ok_rows if r.get("measured_gain_verified")]
    est_gains = [r["estimated_gain"] for r in ok_rows if r["estimated_gain"]]
    if meas_gains:
        ver_part = (f"verified-fused gain geomean: {geomean(ver_gains):.4f}   "
                    if ver_gains else "verified-fused gain geomean: n/a   ")
        print(f"measured gain geomean: {geomean(meas_gains):.4f}   " + ver_part
              + f"estimated gain geomean: {geomean(est_gains):.4f}")
    n_def = sum(1 for r in ok_rows if r["fused_verified"])
    n_forced = sum(1 for r in ok_rows if r.get("fused_verified_forced"))
    print(f"fused (Triton template folded epilogue): default-compile {n_def}/{len(results)}, "
          f"forced-template {n_forced}/{len(results)} configs")
    if err_rows:
        print(f"WARNING: {len(err_rows)} config(s) errored (rows kept in JSON with an 'error' field)")
    print("=" * 130)


if __name__ == "__main__":
    main()
