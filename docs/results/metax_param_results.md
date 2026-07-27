# Does MoE-LLM fusion benefit depend on model parameters? (physical C500)

GLM-5.2 decode fusions gave ~0% on the C500. Is that intrinsic to the algorithmic *pattern* (its
params), and would other parameters help? Swept dim / MoE-vs-dense / batch on the physical C500,
measured the fusion **ceiling** AND the **realized** benefit through the actual cuBLAS/torch.compile
stack. Verified by an adversarial audit that overturned two of my interim claims. Scripts:
`metax_param_sweep.py`, `metax_glm_measure*.py`. Log: `notes/param_sweep_plan.md`.

## Answer: half yes, half no

**YES — GLM-5.2's ~0% is intrinsic to its parameters** (not a C500 quirk). Its wide hidden (6144) +
256 experts + decode batch put it in the **weight-bandwidth-bound MoE corner**, the worst case for
fusion. Measured: the up_gate GEMM runs at **97% of the weight-bandwidth floor** (9.31 ms vs 9.01 ms),
bandwidth-saturated streaming 4+ GB of expert weights with *no spare bandwidth to hide a fused vector
op*. Closed form for that regime: ceiling = `2·tokens/(experts·hidden)` = `2·16384/(256·6144)` =
**2.08%** (and ridge/dim = 158/6144 ≈ 2.6% agrees). The whole decode layer is <2% because weight
reads are ~84% of it. So GLM sits ~10× below where the same fusions could reach on a small/dense
model — the near-zero is a property of *the model*, not the silicon.

**NO — you can't just pick a smaller/denser model to unlock fusion on this stack.** The higher
numbers for small models are **theoretical ceilings that the stock C500 stack never realizes.**

## The theoretical ceiling IS strongly parameter-dependent

| axis | sweep | ceiling |
|---|---|---|
| hidden dim (dense FFN) | 1024 → 8192 | 28% → 20% → 11% → 5% |
| experts (16384 tok, H=2048) | 1 / 8 / 64 / 256 | 16% / 16% / 16% / 7.7% |
| batch (attention+residual) | 256 → 16384 | 7.3% → 2.8% |

Two confirmed laws: **compute-bound ceiling = ridge/dim = 41378/H** (R²=0.996 for H≥2048), and
**weight-bound MoE realizable = 2·tokens/(experts·hidden)**. Parameters that *raise* the ceiling:
smaller hidden, fewer experts, smaller batch — GLM has the opposite of all three.

## But the REALIZED benefit is ~0% (or negative) for EVERY model size

This is the audit's high-severity correction to my interim answer. Measuring the *actual fused
kernels* across the sweep:
- **`torch.compile` SwiGLU-into-GEMM: −63% / −23% / −11% at hidden 512 / 1024 / 2048** (i.e.
  *slower* than unfused — compile/dispatch overhead exceeds the few µs of activation traffic saved).
- **cuBLAS `addmm` residual fusion: −3.6%** (mm 1.937 → addmm 2.008 ms).
- **Best realized anywhere: +2.2%** (MoE E=256 swiglu+down), vs its 7.7% ceiling.

So model parameters move the *ceiling* by ~3–5×, but **do not move the realized benefit** — it's ~0%
everywhere on cuBLAS + torch.compile, because the vendor stack cannot fuse an elementwise
epilogue/prologue into the (grouped) GEMM. Capturing the small-dense ceiling needs a hand-written
**CUTLASS / Triton fused kernel** regardless of model parameters.

## Two corrections to my earlier interim claim

1. **The "28% for small models" was ~67% launch-overhead artifact.** These vector kernels are
   sub-0.1 ms; a measured ~22 µs launch floor dominates them. Overhead-corrected, Axis-1 ceilings
   collapse to ~11 / 12 / 8.5 / 4.6% — H=1024 no longer leads, and the small-vs-large spread shrinks
   from ~5.6× to ~2.5×. (H=1024 was also mislabeled "memory-bound"; it's actually small/latency-bound
   with ~7× spare bandwidth.)
2. **The higher ceiling is not realizable on the stock stack** — measured fused paths are ~0/negative
   for all sizes.

## Bottom line

The null result is **partly the algorithmic pattern** (GLM-5.2 is genuinely the fusion worst-case: a
wide-hidden 256-expert weight-bound MoE, ~2% ceiling) **and mostly the software stack** (no model
size realizes fusion without custom kernels — cuBLAS/torch.compile leave even the small-dense 10–20%
theoretical ceilings on the table, often going negative). Different model parameters would give more
theoretical *headroom*, but **not more realized speedup on the C500's off-the-shelf stack.** The only
way to turn any of it into real wins — for GLM or a smaller model — is custom CUTLASS/Triton fused
epilogue kernels.

## Addendum — actually IMPLEMENTING the fusion (Triton epilogue), not just vendor auto-fusion

The runs above tested vendor *auto*-fusion (cuBLAS `addmm`; default `torch.compile`, which keeps the
vendor GEMM opaque and does NOT fuse the epilogue). That is not the fusion algorithm. The MetaX stack
has **Triton 3.0.0** and `torch.compile mode="max-autotune"`, which emits a **Triton GEMM template
with the SwiGLU/residual epilogue fused in** — the real on-chip-intermediate fusion. Measured
(`metax_fused_triton.py`), fused-vs-eager speedup:

| fusion (Triton epilogue-fused) | speedup |
|---|---:|
| F1 mla_o+residual | 0.76× |
| F4 dense up_gate+SwiGLU, H=2048/4096/6144 | 0.66 / 0.72 / 0.76× |
| F4 small-dense, H=1024/2048 | 0.44 / 0.61× |
| F4 GLM MoE (bmm+SwiGLU) | **0.29×** |
| F5 GLM MoE (SwiGLU+down) | 0.30× (correctness MISMATCH — invalid) |

**The fusion is 1.3–3.4× SLOWER — in every case.** Not because fusion is worthless, but because to
fuse you must abandon the vendor cuBLAS GEMM and use a **Triton GEMM, which on the MetaX C500 is far
slower than the vendor kernel** (GLM MoE: vendor bmm 9.6 ms vs Triton 32.6 ms — 3.4×). The predicted
~2–5% traffic saving is real but dwarfed by this **"fusion tax"** of giving up the vendor GEMM. The
gradient still shows the physics — the tax is smallest for the largest dense GEMM (H=6144, 0.76×,
where the vendor advantage is relatively smaller) and worst for the small/grouped GEMMs — but it never
crosses 1.0.

**Corrected conclusion:** fusion on the C500 fails not (only) because tools "can't fuse," but because
the only fusable GEMM backend (Triton) is much slower than the vendor GEMM. To realize the fusion
benefit you need a fused-epilogue GEMM that matches vendor speed — i.e. a **CUTLASS (MACA-cutlass)
kernel with vendor-quality tuning + fused epilogue**, or a vendor `cuBLASLt`-style fused-epilogue API.
`torch.compile`-default gives the fast GEMM but no fusion; Triton gives fusion but a slow GEMM;
neither wins. (Not yet attempted here: a hand-tuned MACA-CUTLASS fused kernel — the one remaining path.)

## Reproduce
```
conda run -n fusion python metax_param_sweep.py       # ceilings across dim/MoE/batch
conda run -n fusion python metax_fused_triton.py      # REAL Triton epilogue fusion vs eager (all < 1.0x)
```
