# Sim–real gap for MoE-LLM fusion — integrated conclusion (H100 est · C500 measured · RTX 4060 measured)

Ties together the three platforms. The question throughout: does the snowcat-roofline estimator's
predicted fusion benefit (S3-N4-r2f: fold vector ops into weight-amortized GEMM epilogues, ~+6%
decode / +4.6% prefill on H100) **actually materialize on real hardware?**

## The three data points

| platform | role | result |
|---|---|---|
| **H100** | analytical estimate | fusion **+6% (decode) / +4.6% (prefill)** |
| **MetaX C500** | measured (MACA stack) | fusion **≈0% / negative** through the vendor stack; Triton-fused GEMM **3.4× slower** than vendor |
| **RTX 4060** | measured (mature CUDA stack) | **verdict A** — with a *custom* fused kernel the gain is real (**+2.9–4.2% geomean**, fused kernel at **~vendor GEMM speed**); through *stock* tooling the null reproduces (**0.99**). **But** at LOCKED clocks the estimator **over-predicts the magnitude ~2×** (delivered fraction **≈0.41–0.49**) |

The 4060 result is verified sound across two rounds: est-vs-measured single-GEMM calibration holds
(geomean 0.94, in the prior band); every headline number traces to the raw JSON; numerics validated
vs fp32. **Round 2 (locked 1500/5501, the calibration point)** is the clean apples-to-apples test and
the definitive magnitude read: measured verified gain **+4.2%** (drift-clean n=7) vs estimator **+8.6%**
→ **delivered ≈0.49**. Round 1's higher apparent delivered-fraction (~0.78) was a **DVFS artifact** —
the unlocked memory clock (215 vs 168 GB/s) shrank the *predicted* gain toward the measured one; at the
locked calibration bandwidth the ~2× over-prediction is exposed. Verdict A (direction + ranking) is
unchanged; only the magnitude sharpens.

## The verdict: the estimator is roughly right; the C500 null was a *tooling* result, not a *model* result

- **The fusion benefit is real** (verdict A, not B). On the 4060's drift-clean dense configs the
  predicted gain materializes in DIRECTION (measured +4.2% vs predicted +8.6% at locked clocks;
  delivered ≈0.49 — the estimator is ~2× optimistic on MAGNITUDE), and the
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

1. **Estimator accuracy (model):** right on DIRECTION/ranking, but ~2× optimistic on MAGNITUDE — the
   single-GEMM roofline is 72–96% accurate, yet the built fusion delivers only ~0.41–0.49 of the
   predicted gain at the locked calibration point. Trustworthy for *whether* to fuse, loose on *how
   much*. (Round 1's ~0.78 was a DVFS artifact of an unlocked memory clock.)
2. **Realizability (tooling):** the predicted win requires a **custom fused-epilogue kernel** on any
   current stack; off-the-shelf `torch.compile`/cuBLAS realize ~0% (except the residual). This is what
   the C500 measured, and it reproduces on NVIDIA.

So: **the fusion is worth doing and the estimate is credible — but capturing it is a kernel-engineering
task (dual-accumulator, grouped for MoE), not a compiler-flag away, on NVIDIA or MetaX alike.**

## Confidence / caveats

- Round 2 (locked clocks) is the confident read: 7 of 12 configs drift-clean at 1500 MHz (all 6 SwiGLU
  configs clean), so the delivered ≈0.49 / ~2×-over-prediction is on solid footing, not the round-1
  5-of-12 DVFS-limited sample. Verdict B ("gain ≈ 0 with working tooling") is firmly excluded; the
  open question is now the *magnitude* over-prediction, not the sign.
- **T6 now run (round 3, locked clocks)** — the five tests below. It answered the RMSNorm-placement,
  top-k, F6, and merge questions on hardware (see the T6 update section). What remains untested is still
  the **grouped-MoE** SwiGLU fusion at scale (single-expert stands in) and the F6 on-chip kernel (Triton
  OOMs — needs CUTLASS).
- All *GEMM-epilogue* fusion measurements are the **dense / single-expert analog**; a fused *grouped*
  GEMM+SwiGLU kernel exists on no stack, so the real GLM MoE FFN fusion is still untested at scale. The
  *memory-bound* merge fusion (E) is the one piece measured directly and realistically.

## T6 update — the realizability hierarchy (RTX 4060, locked clocks, verified)

The five T6 tests (adversarially verified — every number reproduces from raw JSON, no fabrication)
turn the single "~2× over-prediction" into a **three-tier hierarchy where the realized fraction of the
estimator's prediction is set by the fusion's STRUCTURE, not its size:**

| tier | fusions | realized | why |
|---|---|---|---|
| **Memory-bound** (delivers ~100%) | expert-merge / **r2f** (E) | **1.00–1.07×** the predicted 1.20 | stock `torch.compile` fuses it, **no custom kernel** — no vendor GEMM to lose quality against; estimator was if anything *conservative* |
| **GEMM-epilogue** (delivers ~0.5, can lose) | SwiGLU (D/L2), residual/RMSNorm-prologue (B, A/S5) | b1 **0.49**, b2 **0.83**, SwiGLU **≤0.99 (no clean win)** | needs a custom kernel (**0/16 forced-template folds** → NOT stock-fusable on this build); the hand kernel can't beat cuBLAS at skinny GLM per-expert M → SwiGLU is **neutral-to-negative** at real decode dims |
| **Cross-tile** (unreachable without CUTLASS) | RMSNorm-**epilogue** (A/S3), top-k (C), F6 two-GEMM chain (D/L3) | **0 (no verified fused realization)** | S3 mla_o-epilogue placement **structurally INFEASIBLE** (3.9× over SMEM); top-k dropped (≤ the ~3.5% ceiling); F6 **estimator-only** — Triton OOMs, its 0.259× crossover cliff is a **CUTLASS-gated hypothesis** |

**Corrections this forces:**
- **S3 vs S5 answered on hardware:** the RMSNorm→mla_o *epilogue* (S3) is **not buildable** on 99 KB SMEM; only the up_gate-*prologue* (S5) is — and it nets a *loss* on the wide up_gate host (geomean 0.87), winning only on the skinny N=256 router.
- **SwiGLU fusion does NOT cleanly win at real GLM per-expert dims.** Every drift-clean row is ≤0.99×; the apparent "+10% @ tpe64" was a drift-throttle artifact (the one unclean row) and the grouped cross-check is a loss (0.974). So the earlier +5–7% dense-proxy win was a **large-M artifact** that vanishes at the actual decode `M`=16–64.
- **F6's crossover cliff (1.005→0.259→0.15) exists only in the estimator** — Triton couldn't build the on-chip kernel; realizing/testing it needs CUTLASS.

**Net sim-real law:** the roofline estimator is a trustworthy **upper bound whose realized fraction is a function of fusion class** — ~1.0 (memory-bound), ~0.5 and possibly negative (GEMM-epilogue, kernel-quality-limited), 0 (cross-tile, until a CUTLASS backend exists). "Whether to fuse" is well-predicted; "how much you get" depends on whether a vendor-quality fused kernel exists.
