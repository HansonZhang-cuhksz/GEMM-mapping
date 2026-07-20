# GLM-5.2 decode MoE fusions (F1–F6) on the physical MetaX C500 — measured

Ran the six decode fusions on a real C500 via the CUDA-compatible stack (torch + cuBLAS/`addmm` +
`torch.compile`). Measured components in the `fusion` env; estimator with the C500 model in `area`.
Verified by two workflows incl. adversarial audits (the second forced measuring F3/F5, not just
F1/F4). Scripts: `metax_glm_measure.py`, `metax_glm_measure2.py`. Log: `notes/metax_c500_plan.md`.

## The decode layer is dominated by weight-bound FFN GEMMs

Unfused layer ≈ **16.6 ms**, of which:

| kernel | ms | share | bound |
|---|---:|---:|---|
| up_gate (256-expert bmm) | 9.31 | 56% | **weight-BW** (12.8 GiB / 1.43 TB/s ≈ 9.0 ms floor → 96% of bound) |
| down (256-expert bmm) | 4.63 | 28% | **weight-BW** (6.4 GiB → 4.48 ms floor) |
| mla_o | 1.94 | 12% | **compute** (213 TF/s ≈ 94% of peak) |
| router + residual + rmsnorm + swiglu | 0.76 | 4% | memory |

The FFN GEMMs are **84% of the layer and weight-bandwidth-bound**, so *the entire fusion
opportunity on this layer is the ~4% vector tail.*

## Per-fusion verdict — measured on real silicon (F1, F3, F4, F5), estimated where noted

| fusion | unfused | fused (real stack) | result on C500 |
|---|---|---|---|
| **F1** FA+residual | 1.982 (mm+add) | **2.008** (cuBLAS `addmm`) | **no win** (~1% slower; beta-accumulate path costs ~70 µs > plain mm; even preallocated-out) |
| **F2** FA+resid+RMSNorm | ~2.0 | — (F1 mechanism) | **neutral** — mla_o is compute-bound so the folded vector ops were already ~free |
| **F3** RMSNorm+up_gate | 9.31 (good) / 11.82 (eager) | **9.82** (`torch.compile`) | compile only *rescues the slow eager RMSNorm*; still 0.5 ms above bare up_gate — true prologue fusion **not** achieved |
| **F4** up_gate+SwiGLU | 9.549 | **9.607** (`torch.compile`) | **no win** — Inductor compiles (1 graph, 0 breaks) but cannot fuse a SwiGLU epilogue into the vendor batched GEMM |
| **F5** SwiGLU+down | 4.874 | **4.837** (`torch.compile`) | **0.8%, negligible** — same story; the ~5% ceiling (skip the activated round-trip) needs a custom prologue kernel |
| **F6** full FFN | 14.08 | — | **not a single-block SMEM fusion** (activated[16, 2048] = 64 KiB = the *entire* SMEM budget) **and ~no benefit anyway** (weight-read floor 12.9 ms, eroded by mt× re-reads) |

## Key finding: the analytic small wins do NOT survive the real software stack

On the C500 **none of F1/F3/F4/F5 delivers a meaningful win through cuBLAS or torch.compile.** The
reason is structural, not incidental:
- **F1/F2 (attention side):** mla_o is compute-bound (94% of peak), so its time is fixed and the
  folded residual/RMSNorm were already nearly free; `addmm` even adds a small cost.
- **F3/F4/F5 (FFN side):** the GEMMs are weight-bandwidth-bound and dominate, so eliminating a vector
  kernel caps the gain at ~2–5% — and Inductor/cuBLAS **cannot fuse an elementwise epilogue/prologue
  into the vendor grouped GEMM**, so even that thin ceiling is left on the table. Capturing it needs
  a hand-written CUTLASS grouped-GEMM with a fused epilogue (the C500 has the CUTLASS counterpart, but
  it wasn't authored here).
- **F6 (full FFN):** physically can't be a single-block SMEM-resident fusion on 64 KiB, and offers no
  benefit even in principle.

## C500 vs H100, and a correction to the estimator

| fusion | H100 (est) | C500 (est) | C500 (measured) |
|---|---|---|---|
| F1 | 1.029× FUSE | 0.824× skip | **~neutral (1% slower)** |
| F2 | 1.044× FUSE | 0.827× skip | ~neutral |
| F3 | 1.002× | 1.002× | ceiling ~5%, unrealized |
| F4 | 1.020× | 1.020× | no win |
| F5 | 1.020× | 1.020× | ~5% ceiling, unrealized |
| F6 | 0.515× skip | INFEASIBLE | not-single-block + no benefit |

The estimator's **C500 F1/F2 "skip" (0.82×) is an artifact** — it rests on the mla_o GEMM, which
the C500 model over-predicts 2.58× (the SMEM-tile-cap bias). The real verdict is *neutral*, not an
18% loss; the practical "don't bother" conclusion happens to coincide. The estimator's F3/F4/F5
"1.02× FUSE" is directionally right about the tiny ceiling but blind to the fact that the **standard
stack can't realize it**.

## Bottom line

On the real C500, the trustworthy, hardware-grounded conclusions are: **(1) the decode layer is
weight-bound FFN, so fusion's whole ceiling is ~4%; (2) the off-the-shelf cuBLAS/torch.compile stack
realizes essentially none of it; (3) F6 (full-FFN single-kernel fusion) is out of reach on 64 KiB
SMEM and pointless anyway.** Any real fusion win on C500 requires custom CUTLASS grouped-GEMM
epilogue/prologue kernels — the analytically-predicted 1–4% wins are below what the vendor stack
delivers.

## Reproduce
```
conda run -n fusion python metax_glm_measure.py   metax_glm_measure2.py   # measure F1/F3/F4/F5 on C500
conda run -n area   python -c "import metax_c500_model,fusion_time_estimator as f;from gemm_time_estimator import GPUS;f.run(GPUS['metax-c500'])"
```
