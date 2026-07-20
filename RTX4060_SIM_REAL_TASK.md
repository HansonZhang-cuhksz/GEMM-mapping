# TASK — RTX 4060: measure the sim–real fusion gap (disambiguate the C500 null result)

**Audience:** an engineer or agent on the RTX 4060 machine with access to this repo (`GEMM-mapping/`)
but NOT the originating conversation. This file is self-contained.

**Working dir:** `GEMM-mapping/`. All new files go here. Log your steps in
`notes/rtx4060_worklog.md`.

---

## 0. The one question to answer

The snowcat-roofline **estimator** predicts that fusing the GLM-5.2 MoE decode layer's elementwise
ops into GEMM epilogues (config `S3-N4-r2f`: fold residual/RMSNorm/SwiGLU/residual2, keep the FFN
GEMMs weight-amortized) is **~1–6% faster** than unfused. On a physical **MetaX C500** those wins
**did not materialize (≈0%, often negative)** — but that test was confounded: the MACA software
stack has no working fused-epilogue GEMM path and its Triton backend ran **3.4× slower than the
vendor GEMM**. So we cannot tell whether the fusion benefit is:

- **(A) real but un-capturable on C500** — an immature-tooling problem (fixable with custom
  MACA-CUTLASS), or
- **(B) not real** — the estimator over-predicts fusion benefit (a model problem).

**The RTX 4060 is NVIDIA/CUDA with a mature stack** (cuBLASLt fused epilogues, native CUTLASS,
`torch.compile` max-autotune that emits *competitive* Triton GEMM templates with epilogue fusion),
AND the estimator is **already calibrated on the 4060** (the `RTX4060_MEASURED` profile in
`gemm_time_estimator.py` is validated to 72–96% of measured). So the 4060 can implement the fusion
on a *working* path and measure whether the predicted gain appears — cleanly deciding **A vs B**.

**Deliverable:** a verdict — does the estimated ~1–6% fusion gain materialize on real CUDA hardware
with working tooling? Plus the supporting measurements.

---

## 1. Environment

- A CUDA PyTorch env (≥ py3.11 so the estimator's `snowcat_demo` imports; it uses `enum.StrEnum`).
  Triton ships with recent torch. Verify: `python -c "import torch, triton; print(torch.__version__,
  torch.version.cuda, triton.__version__, torch.cuda.get_device_name(0))"`.
- The estimator is pure-python (`gemm_time_estimator.py`, `fusion_time_estimator.py`) + `snowcat_demo/`.
  If a separate measurement env lacks the estimator, run measurement in the torch env and the
  estimator comparison in any py3.11+ env — they exchange JSON.
- **VRAM: 8 GB.** The full GLM-5.2 MoE does NOT fit (up_gate weights alone ≈ 12.8 GB). Use the
  **scaled/dense configs in §3-T4** (they fit easily and are exactly the regime where the estimator
  predicts the *highest* fusion ceiling, so the best case to test realizability).

---

## 2. Reference material already in the repo

- `gemm_time_estimator.py` — the estimator; `GPUS["rtx4060-measured"]` is the validated 4060 profile
  (18.4 TFLOP/s, 170 GB/s, 24 SM, 99 KiB SMEM/block). `optimal_mapping_by_time(m,n,k,gpu)` -> time.
- `fusion_time_estimator.py` — the 6 GLM fusions (F1–F6) fused-vs-unfused; `run(GPUS["rtx4060-measured"])`
  prints the 4060 estimate. **Prior 4060 estimate: F1 1.020×, F2 1.026×, F3 1.002×, F4 1.020×,
  F5 1.020× (all FUSE); F6 0.259× (SKIP).**
- `metax_glm_measure.py`, `metax_fused_triton.py` — the C500 measurement scripts. **They are generic
  `torch.cuda` code and run on the 4060 as-is** — just SCALE THE DIMS DOWN to fit 8 GB (see §3-T4).
  Reuse their timing methodology (bf16, warmup, `torch.cuda.Event`, median).
- `fusion_configs.py` / `fusion_configs_results.md` — the 40-config decode study (H100 estimate:
  best `S3-N4-r2f` +6% over unfused).
- `metax_c500_results.md`, `metax_glm_results.md`, `metax_param_results.md` — the C500 findings to
  contrast against.

---

## 3. Tasks

### T1 — Environment + device discovery
`nvidia-smi`; torch device props (`torch.cuda.get_device_properties(0)`: SM count, L2, SMEM/block,
total_memory, clock). Record in the worklog. Confirm 8 GB and note the SMEM/block (should be ~99–100
KiB opt-in).

### T2 — Confirm the measured peak specs (validate the `rtx4060-measured` profile)
Measure peak BF16 GEMM TFLOP/s (large square GEMMs that fit 8 GB, e.g. 4096³, 6144³) and HBM
bandwidth (large copy). Compare to the profile's 18.4 TFLOP/s / 170 GB/s. **Note:** the 4060 DVFS
(boost vs locked clocks) matters — record whether clocks are locked; the profile was calibrated at
locked 1500/5501 MHz. Adjust the profile via `dataclasses.replace` if your clocks differ.

### T3 — Single-GEMM estimator validation (est vs measured)
Adapt `metax_measure.py`+`metax_compare.py` (or write fresh): measure real bf16 times for a spread of
GEMM shapes (squares 1024–6144; a few tall/wide; the FFN-stage shapes), compare to
`optimal_mapping_by_time(...,GPUS["rtx4060-measured"])`. Report est/meas geomean and % within 1.5×/2×.
**Expected: within the 72–96% band** (the profile was calibrated here). If it's far off, recalibrate
before trusting T4.

### T4 — **THE KEY EXPERIMENT: fusion realizability on working NVIDIA paths**
For each fusion primitive, measure **unfused vs each real fused path**, on **scaled configs that fit
8 GB**, and compare the realized gain to the estimator's prediction for the same dims.

**Scaled configs (all fit 8 GB — pick a small sweep):**
- Dense FFN (the high-ceiling regime): `M ∈ {2048, 8192}`, `hidden ∈ {1024, 2048, 4096}`,
  intermediate = hidden. up_gate `[M, hidden]@[hidden, 2·inter]`, down `[M, inter]@[inter, hidden]`.
- Scaled MoE (optional): `experts ∈ {8, 32}`, hidden 2048, tokens/expert = 64–256 (grouped `bmm`).
- Attention proj (F1): `mla_o = [M, 6144]@[6144, 16384]` for `M ∈ {2048, 8192}` (fits: weights 192 MB).

**Fusion primitive 1 — SwiGLU into the up_gate epilogue (this is `N4`/`F4`, the crux):**
```
def swiglu_ffn(x, Wug):            # x:[M,hidden]  Wug:[hidden,2*inter]
    gu = x @ Wug
    g, u = gu[..., :inter], gu[..., inter:]
    return torch.nn.functional.silu(g) * u      # -> [M, inter]  (half-width activated)
```
Measure THREE things and compare:
  a. **unfused** eager: `gu = x@Wug` (vendor GEMM) then a separate SwiGLU kernel.
  b. **torch.compile(mode="max-autotune")** of `swiglu_ffn` — on NVIDIA this emits a Triton GEMM
     TEMPLATE with the SwiGLU fused into the epilogue (unlike default compile, which leaves the vendor
     GEMM opaque; unlike C500, the NVIDIA Triton GEMM is competitive). Verify it actually fused (check
     it's ~1 kernel and near vendor-GEMM speed, not 3× slower).
  c. **(optional) a hand Triton GEMM+SwiGLU-epilogue kernel** for an upper bound.
  Report fused/unfused speedup for each path. Also confirm numerics match (allclose, bf16 rtol~2e-2).

**Fusion primitive 2 — residual into the mla_o epilogue (`F1`):**
`torch.addmm(residual, x, Wo)` (cuBLAS β·C accumulate, one kernel) vs `(x@Wo)+residual` (two kernels).
Report speedup. Also try `torch.compile` max-autotune of `x@Wo+residual`.

**For each measured (unfused, fused) pair, also compute the estimator's predicted gain** for the same
dims via `fusion_time_estimator` (adapt its `estimate_fused_gemm`/`estimate_gemm_grouped` with the
scaled dims + `GPUS["rtx4060-measured"]`), so you have **measured-gain vs estimated-gain side by side.**

### T5 — Verdict
Aggregate T4 into the A-vs-B decision (see §5). Write `rtx4060_sim_real_results.md`.

---

## 4. Deliverables (write these files to `GEMM-mapping/`)

1. `notes/rtx4060_worklog.md` — step log (T1–T5).
2. `rtx4060_measured.json` — T2/T3 raw measurements.
3. `rtx4060_fusion.json` — T4 raw: per config, {unfused_ms, fused_ms (per path), est_unfused_ms,
   est_fused_ms, measured_gain, estimated_gain, fused_verified_bool}.
4. `rtx4060_sim_real_results.md` — the writeup: the est-vs-measured fusion-gain table, whether the
   fusion FUSED on the working path (kernel count / speed), and the **A-vs-B verdict**.

---

## 5. Decision criteria (what the answer means)

Let `g_est` = estimator's predicted fusion gain (≈ +1–2.6% for F1/F4/F5 on the 4060 profile),
`g_meas` = the best measured gain across the working fused paths.

- **If `g_meas` ≈ `g_est` (fusion genuinely faster, and the Triton/cuBLASLt fused kernel runs at
  ~vendor-GEMM speed):** → **verdict A** — the fusion algorithm is SOUND; the C500 null was a
  *tooling* problem (immature MACA Triton / no cuBLASLt epilogue). Actionable: a custom
  MACA-CUTLASS fused-epilogue kernel would recover the win on C500.
- **If `g_meas` ≈ 0% or negative EVEN THOUGH the fused kernel ran at competitive GEMM speed:**
  → **verdict B** — the estimator OVER-PREDICTS fusion benefit; the eliminated vector traffic is
  already hidden/negligible on real hardware. Fusion isn't worth it, C500 or not.
- **If the fused path itself was slow (Triton GEMM ≫ cuBLAS, like C500):** → inconclusive on NVIDIA
  (shouldn't happen — flag it), but it would mean the tooling story generalizes.

Also report **which fused PATH worked** (max-autotune Triton template vs cuBLASLt `addmm` vs hand
Triton) — that tells us what a production kernel would need.

---

## 6. Numbers to check against (from the H100/C500 study; expect the 4060 to sit between)

| quantity | H100 (estimate) | C500 (measured) | 4060 (estimate; to be measured by you) |
|---|---|---|---|
| S3-N4-r2f fusion gain, decode | +6% | ≈0% / negative | est +1–2.6% per-fusion → **measure it** |
| F1 residual (cuBLAS addmm) | small win | 2.001 vs 1.982 ms (1% slower) | measure |
| F4 SwiGLU→up_gate (compile) | +2% | 9.607 vs 9.549 ms (no fuse) | **measure — does max-autotune fuse & win?** |
| Triton fused GEMM vs vendor | — | 3.4× SLOWER (fusion tax) | **measure — should be ~competitive on NVIDIA** |
| SMEM/block | 227 KiB | 64 KiB | ~99 KiB |

The single most informative datapoint: **does `torch.compile(mode="max-autotune")` on the SwiGLU-FFN
produce a fused Triton GEMM that (a) runs at ~vendor-GEMM speed and (b) beats the unfused path?** On
C500 the answer was no on both counts. On the 4060 it should be yes on (a); (b) is the open question.

---

## 7. Caveats to honor

- bf16, warmup ≥ 15 iters, median of ≥ 30, `torch.cuda.synchronize()` around `Event` timing; fixed
  device 0; lock clocks if possible (or record them).
- Verify every "fused" kernel actually fused (kernel count via a profiler or by checking it's not
  falling back to eager) and is numerically correct — the whole point is defeated if compile silently
  runs eager.
- Keep everything ≤ 8 GB; prefer the dense/small-hidden configs (also the highest-ceiling regime).
- These are **relative** (fused/unfused) measurements — robust to absolute-calibration error, which is
  what we want.
