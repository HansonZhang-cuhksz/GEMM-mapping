# Sim–real gap for MoE-LLM fusion — integrated conclusion (H100 est · C500 measured · RTX 4060 measured)

Ties together the three platforms. The question throughout: does the snowcat-roofline estimator's
predicted fusion benefit (S3-N4-r2f: fold vector ops into weight-amortized GEMM epilogues, ~+6%
decode / +4.6% prefill on H100) **actually materialize on real hardware?**

## The three data points

| platform | role | result |
|---|---|---|
| **H100** | analytical estimate | fusion **+6% (decode) / +4.6% (prefill)** |
| **MetaX C500** | measured (MACA stack) | fusion **≈0% / negative** through the vendor stack; Triton-fused GEMM **3.4× slower** than vendor |
| **RTX 4060** | measured (mature CUDA stack) | **verdict A** — with a *custom* fused kernel the gain is real (**+3.7% geomean**, ~78% of the +4.6% predicted, fused kernel at **~vendor GEMM speed**); through *stock* tooling the null reproduces (**0.992**) |

The 4060 result is verified sound: est-vs-measured single-GEMM calibration holds (geomean 0.937, 96%
within 1.5×, in the prior band); every headline number traces to the raw JSON; numerics validated vs
fp32; DVFS/throttling handled by drift probes + contemporaneous-GEMM referencing.

## The verdict: the estimator is roughly right; the C500 null was a *tooling* result, not a *model* result

- **The fusion benefit is real** (verdict A, not B). On the 4060's drift-clean dense configs the
  predicted gain materializes (+3.7% measured vs +4.6% predicted; delivered fraction ~0.78), and the
  genuinely-fused kernel runs at ~0.99× the vendor GEMM — **no fusion tax on a mature stack.** So the
  estimator over-predicts modestly (~20%) but is directionally and roughly magnitude-correct.
- **The C500 was not measuring wrongly** — it correctly found that its *vendor stack* can't fuse.
  The 4060 shows the same thing happens on NVIDIA's *stock* stack (0.992 all-configs): off-the-shelf
  `torch.compile` / cuBLAS capture ~0% of the SwiGLU fusion.

## The sharpened, universal finding: SwiGLU fusion needs a *custom* kernel on **any** stack

The 4060 pinned down *why* stock tooling fails, and it's not MetaX-specific:
- **Residual fusion IS stock-fusable** on CUDA — `torch.addmm` (cuBLASLt β-accumulate) captures it
  (+0.6…+3.5%). The C500's addmm-is-slower was a genuine vendor-library difference (tooling).
- **SwiGLU fusion is NOT stock-fusable, even on NVIDIA** — two reasons: (1) Inductor gates Triton
  GEMM templates to big GPUs (≥68 SMs; the 4060 has 24, the C500 similar), and (2) *structurally*,
  `silu(gate)·up` combines two disjoint column-slices from **different output tiles**, which a
  template epilogue (elementwise-on-own-tile) cannot express. Only a **dual-accumulator hand kernel**
  (Triton here, CUTLASS on C500) collects the gain (+5.6…+7.4% at vendor speed).
- The C500's specific **3.4× Triton tax does NOT generalize** (NVIDIA Triton GEMMs run 0.96–1.59×);
  but the "stock tooling → ~0% fusion" conclusion **does** generalize.

## What is still unconfirmed — the real GLM MoE decode regime

The actual GLM-5.2 decode FFN is a **grouped (per-expert) MoE**. **No fused grouped-GEMM+SwiGLU
kernel exists on *either* stack** — the hand Triton kernel is dense-only, and Inductor's bmm template
can't fold SwiGLU. So the estimator's **+6% for the real grouped MoE is demonstrated only in the
dense analog, not on the true workload.** Confirming it needs a custom *grouped* fused-epilogue kernel
that doesn't exist anywhere yet.

## Bottom line — the sim–real gap has two independent layers

1. **Estimator accuracy (model):** roughly right — 72–96% on single GEMMs; the +6% fusion prediction
   delivers ~78% when actually built. The roofline model is trustworthy for relative fusion verdicts.
2. **Realizability (tooling):** the predicted win requires a **custom fused-epilogue kernel** on any
   current stack; off-the-shelf `torch.compile`/cuBLAS realize ~0% (except the residual). This is what
   the C500 measured, and it reproduces on NVIDIA.

So: **the fusion is worth doing and the estimate is credible — but capturing it is a kernel-engineering
task (dual-accumulator, grouped for MoE), not a compiler-flag away, on NVIDIA or MetaX alike.**

## Confidence / caveats

- 4060 verdict rests on 5 of 12 drift-clean configs (WSL2, unlockable clocks + thermal throttling on a
  35 W part); the machine's sensitivity analysis brackets it (+3.8/+4.8% adding back tainted-positive
  rows; only counting an unmeasurable-baseline row erodes it to +1.4/+6.0%). Moderate, not high,
  confidence — but verdict B ("gain ≈ 0 with working tooling") is firmly excluded.
- All fusion measurements are the **dense analog**; the grouped MoE (the real workload) is untested on
  hardware. This remains the key open experiment.
