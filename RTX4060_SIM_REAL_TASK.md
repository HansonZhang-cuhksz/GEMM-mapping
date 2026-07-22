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

---

# T6 — RMSNorm + MoE-structure fusion tests (update)

*Appends to this file. Extends the T4 fusion study (`rtx4060_fusion_measure.py`, `rtx4060_common.py`, `fusion_time_estimator.py`) with five new tests (A–E) covering the RMSNorm placements and the MoE-structure fusions (router prologue, top-k epilogue, per-expert FFN levels, expert-merge). Every test reuses the T4 methodology verbatim — see the shared preamble — and every numeric figure quoted in a design cell below is an **estimator-derived prediction, not a measurement**, unless a cell is explicitly labeled "measured".*

## T6.0 — Shared preamble (read once, applies to A–E)

### T6.0.1 GLM-5.2 params (EXACT — match the repo constants)

`HIDDEN = 6144`, `INTERMEDIATE = 2048`, `2*INTERMEDIATE = 4096`, `EXPERTS = 256`, `TOP_K = 8`, `KV = 16384`, `BPE = 2` (bf16, fp32 accumulate), `eps = 1e-6`. Draw `gamma` once (`randn(HIDDEN)*0.1 + 1`) and keep it identical between baseline and fused.

**Which M each op uses (do not confuse these):**
- **Dense-over-all-tokens** ops — `mla_o`, `router`, expert-`merge`: `M = tokens` directly.
- **Per-expert** ops — `up_gate`, `down`: `M = tpe = tokens/32` (= `tokens · TOP_K/EXPERTS = tokens·8/256`).

### T6.0.2 Single-expert modeling + caveats (confirmed)

Under balanced routing all 256 experts are identical-shape GEMMs, so **run ONE expert at `M = tpe`** and take layer cost ≈ `256 × one-expert`; the **fused/unfused ratio** is the representative quantity. Caveats to state in every writeup that uses it:
- **(a)** assumes balanced routing — real load-imbalance/stragglers are orthogonal to fusion.
- **(b)** the cross-expert **merge/combine** is NOT covered by single-expert modeling — it is modeled separately as a synthetic dense TOP_K=8 merge (Test E), never as one expert.
- **(c)** these are STANDALONE GEMM/vector microbenchmarks — no O(T²) attention core; "prefill" just means large `M`/`tpe`. In the real layer the residual stream (`h`) and normalized hidden (`x`) are shared downstream, so the true op-attributable saving is smaller than the standalone microbenchmark shows (state this wherever a shared tensor is involved — B, A-prologue).

### T6.0.3 Batch regimes (both required, every test)

| regime | `tokens` | `tpe = tokens/32` (per-expert ops) | `mt = tpe/16` (F6 row-blocks) |
|---|---|---|---|
| DECODE | 512, 1024, 2048, 4096, 8192, 16384 | 16, 32, 64, 128, 256, 512 | 1, 2, 4, 8, 16, 32 |
| PREFILL | 8192, 32768, 131072 | 256, 1024, 4096 | 16, 64, 256 |

(`tokens=8192 → tpe=256` is the decode/prefill boundary; keep it in both — it anchors the regime crossover.) Per-test memory caps override the top prefill point where noted (A epilogue ≤ 32768; E merge drops 131072, see E). All non-capped configs FIT 8 GB.

### T6.0.4 Reuse the 6-path taxonomy + methodology (verbatim from T4)

Six paths per config: `unfused | addmm | compiled | nocg | forced | triton`.
- **Timing:** `rtx4060_common.med_time` — median-of-30 CUDA events after ≥15 warmup, busy-GEMM clock warmer (`warm_clocks`), `ClockSampler`, `sleep_cooldown`, `geomean`, `save_json`.
- **Drift probe:** re-measure the bare reference kernel (bare GEMM, or bare reduction for E) at config END; clean iff `|drift−1| ≤ 0.05`. Drift-tainted rows are kept in JSON but excluded from the aggregate/geomean.
- **Fused verification:** `profile_kernels` + `judge_fused`/`vendor_fused_ok`; store raw `kernel_evidence` per path. A path counts as fused only if the profiler shows the epilogue/prologue/reduction folded with no surviving separate kernel.
- **Numerics:** fp32 ground-truth per config; each path OK iff `rel_max_vs_fp32 ≤ max(2·eager_rel_max, 5e-2)` (fp32-accumulated fused kernels are typically ≥ eager-accurate, so allclose-vs-eager is the wrong test).
- **Metrics:** `measured_gain = best_unfused_ms / best_fused_ms` (C500-convention, includes unverified paths) AND the verdict metric **`measured_gain_verified = best_unfused_ms / best VERIFIED-fused path`**. Also report `est_unfused_ms / est_fused_ms / estimated_gain` (stock + `--t2-json` adjusted profile via `build_adjusted_profile`), the fused-kernel ÷ contemporaneous `gemm_only` fusion-tax ratio, cudagraph static-input-copy µs, and the geomean over drift-clean configs only.
- **Structure-blind reporting guard (applies to A-epilogue, C):** `estimate_fused_gemm` models the epilogue as **tile-local only** (`Epilogue` exposes `a_factor / out_factor / extra_hbm_once / aux_smem_per_tile` — confirmed no cross-N-reduction and no selection/sort term). Any "fuse" number it emits for a **cross-tile-reduction** epilogue (RMSNorm-over-N, top-k) is **NOT a prediction** — tabulate that cell as `est=INVALID (structure-blind)` and **exclude it from all delivered-fraction / est-vs-measured aggregate stats**; present it only as the standalone est-vs-measured structural-mismatch datapoint.

### T6.0.5 Deliverables convention

Each test names a new measure function + writes rows to a JSON and a results sub-section (see per-test Deliverables and the consolidated T6 Deliverables note at the end).

---

## T6.A — RMSNorm fused into `mla_o` (epilogue, structurally infeasible) + prologue fallback

New script `rtx4060_rmsnorm_fuse.py` (modeled on `rtx4060_fusion_measure.py`); raw → `rtx4060_rmsnorm.json`.

### A.1 The fusion + exact dims

Post-attention RMSNorm (`y[i,:] = x[i,:]·gamma / sqrt(mean_j x[i,j]² + eps)`) reduces over the **full `HIDDEN=6144` row**. Two placements:
- **Epilogue (primary):** fold the norm into the `mla_o` output projection `[tokens, HIDDEN, KV] = [tokens, 6144, 16384]`, `K=KV=16384`, `N=HIDDEN=6144`. The norm reduces over `N` (split across CTAs).
- **Prologue (fallback):** fold the norm into the input-load of the next GEMM. Primary host `up_gate` (per-expert) `[tpe, 4096, 6144]`; secondary host `router` (dense) `[tokens, 256, 6144]`. Both reduce over `K=HIDDEN` (streamed inside ONE CTA).

**Regimes.** DECODE: epilogue `M=tokens ∈ {512..16384}`; prologue `up_gate M=tpe ∈ {16..512}`. PREFILL: prologue `up_gate M=tpe ∈ {256,1024,4096}`. **Epilogue memory cap:** at `tokens=131072`, `mla_o` A `[131072,16384]`=4.0 GiB + out `[131072,6144]`=1.5 GiB + a normalized copy exceed 8 GiB → **cap `mla_o`-epilogue prefill at `tokens ≤ 32768`, OR process M in row-chunks of ≤32768** (the norm reduction is M-independent, so one chunk is representative — state which you did). Residual coupling: the real S3 input is `mla_o_out + residual1`; residual-add is tile-local (F1) and not the difficulty — this test isolates the reduction (optionally add a pre-materialized `res` into the input to mirror the layer; note which). The verdict keys on the reduction.

### A.2 Explicit unfused baseline (spec output requirement 2)

Two kernels, best-of-eager/compiled:
- **Epilogue:** `out = mla_o(x)` (cuBLAS bf16 GEMM, raw write) then a **separate** RMSNorm reduction kernel `y = F.rms_norm(out, (HIDDEN,), gamma)` (reads `out`, writes `y`; vector traffic ≈ `2·M·HIDDEN·BPE`). **`best_unfused = min(eager, nocg)`** where `nocg` = vendor GEMM + one fused reduction kernel (max-autotune-no-cudagraphs).
- **Prologue:** separate RMSNorm kernel `xn = rms_norm(x)` (reads `h`, writes `xn`) then `up_gate(xn)`. **`best_unfused = min(eager, nocg)`.**

### A.3 6-path verdict

**Epilogue (reduction over OUTPUT dim N, split across CTAs — cross-tile, strictly harder than SwiGLU which combines only 2 column-slices):** `unfused` = baseline; `addmm` = **DROP** (cuBLASLt β-accumulate is tile-local element-wise, no RMSNorm/reduction epilogue in torch's binding — record `addmm_ms=null`, reason logged); `compiled`/`nocg`/`forced` = run + profile, expected to leave a **separate** reduction kernel (Inductor template epilogues are tile-local elementwise; 24 SM `is_big_gpu`<68 declines the Triton GEMM too) — verify via `judge_fused`; `triton` (hand wide-tile) = **attempt, expect infeasible → drop** (A.4).

**Prologue (reduction over CONTRACTION dim K, tile-local to the CTA's K-loop):** `unfused` = baseline; `addmm` = N/A; `compiled`/`nocg`/`forced` = run + profile, expected to NOT eliminate the normalized-A round-trip (separate normalize kernel writes full normalized A, then vendor GEMM re-reads); `triton` (hand kernel) = **the realizable fused path** (A.4).

### A.4 Custom kernel

**Epilogue — three options, all fail on 99 KiB SMEM (`smem_per_block_bytes = 101376`); enumerate, attempt option 1, drop:**
1. **BN=full-HIDDEN wide tile** (one CTA owns the `[BM, 6144]` row → reduction CTA-local). fp32 stage = `BM·6144·4` B. At MMA floor `BM=16` that is **384 KiB — 3.9× over the ~99 KiB budget** (393216/101376 = 3.88); even `BM=4` ≈ 98 KiB leaves nothing for K-streaming and wrecks tensor-core efficiency. **Infeasible.**
2. **Two-pass epilogue within the CTA** — applying the scale needs the sum-of-squares after the full-N pass, so each output tile's MMA over `K=16384` is recomputed → **2× the entire GEMM compute**; `mla_o` is compute-heavy, doubling FLOPs erases any saving. **Loses.**
3. **Split-K partial-sum + atomics** — the raw `[M,6144]` output is still written to HBM and read back to rescale → the RMSNorm round-trip (the thing fusion removes) is **not saved**, only a launch. **Marginal, not a real fusion.**
> Instruction: implement option 1 at smallest `BM`, confirm the SMEM assert fires (or compile falls back), record `smem_per_block_bytes` + the `BM·HIDDEN·4` budget, keep the row with an `infeasible_reason` field, and **document the epilogue as structurally infeasible on 99 KiB SMEM, then fall back to the prologue.** This is a result: the tile-local estimator (A.5) predicts a fuse; the hardware says no.

**Prologue — the realizable design (reduction over K is tile-local to the main loop):**
- **P2 (recommended, matches estimator F3):** kernel 1 reads `A=[M,6144]`, reduces sum-of-squares over K (fp32), writes `inv_rms=[M]`; kernel 2 = standard tiled `up_gate` GEMM whose prologue multiplies each loaded A-tile by `inv_rms[row]` and `gamma[k]` before the MMA. Saves the write+reread of the full normalized `[M,6144]`; costs one extra cheap read of A. `aux_smem = BM·4` (+ BK gamma slice). `a_factor≈1.0`.
- **P1 (secondary):** single-kernel two-pass-over-K, reads A twice (`a_factor≈2.0`); acceptable in weight-bound decode.
- Grid `(cdiv(M,BM), cdiv(N,BN))`, `N=4096` (up_gate) or `256` (router), `K=6144`; mini tile-autotune over `TRITON_CANDS`. Numerics per T6.0.4.

### A.5 Estimator wiring + structure-blind flag

Add `est_rms_epilogue_ms(M, hidden, kv, gpu)` and `est_rms_prologue_ms(M, twoI, hidden, count, gpu, a_factor)` mirroring `est_swiglu_ms`/`est_residual_ms` (import `estimate_gemm_grouped`, `estimate_fused_gemm`, `estimate_vector_kernel`, `Epilogue`, `BPE`; use `GPUS["rtx4060-measured"]` + `--t2-json`):

```python
RMS_STAT = 4  # fp32 per row
def est_rms_epilogue_ms(M, hidden, kv, gpu):        # STRUCTURE-BLIND — see below
    unf_g = estimate_gemm_grouped("mla_o", M, hidden, kv, 1, gpu)
    unf_r = estimate_vector_kernel("rmsnorm", 2*M*hidden*BPE, gpu)     # read out + write normalized
    fus   = estimate_fused_gemm("mla_o+rms_epilogue", M, hidden, kv, 1,
              Epilogue(aux_smem_per_tile=lambda m0,n0: m0*RMS_STAT + n0*BPE), gpu)  # gamma+stat
    return (unf_g.time_s+unf_r.time_s)*1e3, fus.time_s*1e3
def est_rms_prologue_ms(M, twoI, hidden, count, gpu, a_factor=1.0):    # F3-analog, faithful
    unf_g = estimate_gemm_grouped("up_gate", M, twoI, hidden, count, gpu)
    unf_r = estimate_vector_kernel("rmsnorm", M*hidden*BPE + M*RMS_STAT, gpu)  # read h, write stat
    fus   = estimate_fused_gemm("up_gate+rms_prologue", M, twoI, hidden, count,
              Epilogue(a_factor=a_factor, aux_smem_per_tile=lambda m0,n0: m0*RMS_STAT), gpu)
    return (unf_g.time_s+unf_r.time_s)*1e3, fus.time_s*1e3
```
`a_factor=1.0` for P2, `2.0` for P1.

**REPORTING GUARD (mandatory):** `estimate_fused_gemm` has **no cross-N-reduction term** (the `Epilogue` dataclass only exposes tile-local per-operand factors + aux SMEM — confirmed in `fusion_time_estimator.py:158–163`), so `est_rms_epilogue_ms` will emit a bogus "fuse" verdict for the epilogue. In the results table mark the epilogue est cell **`est=INVALID (structure-blind)`** — it is **not a prediction** and must be **excluded from delivered-fraction/est-vs-measured aggregate stats**; present it only as the standalone structural-mismatch datapoint contrasting the measured infeasibility. The prologue's F3 model is faithful (reduction is legitimately K-local) — `est_rms_prologue_ms` is the valid yardstick there.

### A.6 Expected outcome (all figures below are estimator-predicted, not measured)

- **Epilogue:** no stock path fuses (`addmm` N/A; `compiled`/`nocg`/`forced` leave a separate reduction — verify); the hand wide-tile fails the SMEM budget (3.9× over). → **Structurally infeasible on this hardware**; document with the SMEM math + failed-attempt evidence, drop. The structure-blind est (flagged) predicting a fuse is the contrast.
- **Prologue:** stock paths do not remove the normalized-A round-trip (verify); the hand **P2** (or P1) is the only genuinely-fused path, **predicted** to run ~0.98–1.05× the bare `up_gate` GEMM (matching the SwiGLU hand-kernel result) and deliver a small **predicted** positive gain (F3-class ~+1–2% in decode where `up_gate` is weight-bound, shrinking toward compute-bound parity in prefill). Report `measured_gain_verified` vs `estimated_gain` (F3-analog); **expect** delivered-fraction < 1, consistent with T4.
- **Verdict:** RMSNorm confirms the §4 lesson — cross-tile-reduction fusions are unreachable by stock tooling and, for the epilogue placement, by any kernel on 99 KiB SMEM; the prologue relocates the reduction onto the next GEMM's K-loop where a custom kernel fuses it (generalizes to a C500 MACA-CUTLASS prologue kernel).

---

## T6.B — router PROLOGUE fusion (b1 residual+RMSNorm, b2 residual-only)

New script `rtx4060_router_prologue.py`; raw → `rtx4060_router_prologue.json`.

### B.1 The fusion + exact dims

Router = dense GEMM `logits[M,256] = X[M,6144] @ Wrouter[6144,256]` (`m=M=tokens`, `n=EXPERTS=256`, `k=HIDDEN=6144`). Its input `X` is produced by `h = attn_out + residual_in` then `x = RMSNorm(h)`. A prologue folds these into the router GEMM's A-load.
- **(b2) residual-only:** fold `h = attn_out + residual_in` (`logits = h @ Wrouter`).
- **(b1) residual + RMSNorm:** fold `h`, the per-row `rms`, and `gamma` (`logits = RMSNorm(h) @ Wrouter`).

Dims (router is dense → `M = tokens`, NOT tokens/32): DECODE `M ∈ {512..16384}`, PREFILL `M ∈ {8192,32768,131072}`; `k=6144`, `n=256`.

**Absolute-magnitude caveat (state prominently):** the router is a minor cost center — `n=256` is tiny, FLOPs `2·M·256·6144` are ~16× below one up_gate expert-layer and dwarfed by `mla_o`. A healthy *relative* prologue speedup is a *small absolute* layer saving. Second caveat (T6.0.2c): in the real layer `h` and `x` are shared downstream, so the router-*attributable* saving is only the avoided re-read of `x` — smaller than this standalone microbenchmark shows. Note both explicitly.

### B.2 Explicit unfused baseline

- **b2:** eager `residual` kernel then plain router GEMM (2 kernels). **`best_unfused = min(eager, nocg)`**, `nocg` = vendor GEMM + one fused pointwise add.
- **b1:** eager `residual` then eager `RMSNorm` (reads `h`, writes `x` + tiny stat) then plain router GEMM (3 kernels). **`best_unfused = min(eager, nocg)`.**
- Drift probe = bare `X @ Wrouter` re-measured at config end.

### B.3 6-path verdict

`unfused` = baseline. `addmm` = **DROP** (β-accumulate adds a `[M,256]` tensor to the *output*; a prologue residual is on the `[M,6144]` *input* at a different stage — structurally inexpressible; a clean contrast to F1 where addmm was the star). `compiled` = expect no fuse (24-SM `is_big_gpu` gate keeps vendor cuBLAS + separate add; RMSNorm's K-reduction never folded). `nocg` = best-unfused realization, not a fused candidate. `forced` = the key stock test: **b2** the residual add is tile-local on the input → if inductor prologue-fusion fires it folds into a single `triton_tem_fused_*` (verify kernel evidence); **b1** even forced, inductor cannot co-accumulate `sum_k h²` inside the matmul template → at best folds residual, leaves RMSNorm as a separate reduction kernel. `triton` (hand) = guaranteed-fused upper bound (b2) / **only fully-fused path** (b1).

**Feasibility crux.** Residual prologue is tile-local (each `A[i,k]` gets its own `res[i,k]`; no cross-k dependency). RMSNorm prologue needs the input-row reduction over full `K=6144`, single-pass fusable only in a hand kernel via: (1) `gamma` is a per-k factor on the contracted axis → **folds into the weight** `W' = diag(gamma)·Wrouter` (precompute once); (2) `1/rms[i]` is a per-row epilogue scale. A hand kernel co-accumulates `sum_k h[i,k]²` in the SAME K-loop (A-tile already resident) → `rms` free at K-loop end, applied as `logits[i,j]=acc[i,j]/rms[i]`. No compiler synthesizes this fold.

### B.4 Custom Triton kernel

Needed: yes for b1 (only fully-fused path), yes for b2 as the guaranteed upper bound / fallback if inductor prologue-fusion does not fire. Router `m=M, n=256, k=6144`; both fit SMEM trivially (n tiny, one fp32 `sumsq[BM]` vector). b2: `acc += tl.dot(a + r, W[k,bn])`. b1 dual-accumulator single-pass:

```
precompute (host, untimed):  Wp = Wrouter * gamma[:, None]        # diag(gamma)·W
per CTA (owns [BM, BN=256], full K):
    acc = zeros(BM,256); sumsq = zeros(BM)                        # fp32
    for k0 in range(0, K, BK):
        h = load A[bm,k0:k0+BK] + load R[bm,k0:k0+BK]             # residual prologue (tile-local)
        sumsq += tl.sum(h.to(fp32)*h.to(fp32), axis=1)           # reduction, free (h resident)
        acc   += tl.dot(h, Wp[k0:k0+BK, bn])                     # gamma folded into Wp
    rms = tl.sqrt(sumsq / K + eps)
    store logits[bm,bn] = acc / rms[:, None]
```
Grid `(cdiv(M,BM), 1)` (`BN=256`), `BM ∈ {64,128}`, `BK ∈ {32,64}`, warps 4–8, stages 2–3, mini-autotune. **HARD CONSTRAINT — no split-K** (single-pass `sumsq` is only correct if one CTA owns the full K-loop for its rows; `M/BM` CTAs give ample parallelism; enforce it and let fp32 numerics catch any violation). Numerics per T6.0.4.

### B.5 Estimator wiring (corrected `a_factor` labeling)

`gain = unfused/fused`. **The residual's `a_factor=2.0` is the RESIDUAL's second `[M,HIDDEN]` operand read** (attn_out + residual_in = one extra A-worth of read), NOT F5's gate+up 2×-wide contraction and NOT the RMSNorm — RMSNorm adds NO extra A-read (it is the free K-reduction over the already-resident A-tile). RMSNorm's cost is the per-row K-reduction aux (`m0·4` fp32 partials, repo F3) only. Keep these separable so the A-read is counted once for the norm:

```python
# b2 residual-only prologue: a_factor=2.0 = residual second-operand read
def est_router_residual_ms(M, gpu):
    g = estimate_gemm_grouped("router", M, EXPERTS, HIDDEN, 1, gpu)
    r = estimate_vector_kernel("residual", 3*M*HIDDEN*BPE, gpu)       # read attn, read res, write h
    f = estimate_fused_gemm("router+res_prologue", M, EXPERTS, HIDDEN, 1,
                            Epilogue(a_factor=2.0), gpu)              # residual second-operand read
    return (g.time_s + r.time_s)*1e3, f.time_s*1e3

# b1 residual + RMSNorm prologue: a_factor=2.0 (residual read) + aux m0*4 (rmsnorm K-reduction)
def est_router_residual_rms_ms(M, gpu):
    g = estimate_gemm_grouped("router", M, EXPERTS, HIDDEN, 1, gpu)
    r = estimate_vector_kernel("residual", 3*M*HIDDEN*BPE, gpu)
    n = estimate_vector_kernel("rmsnorm",  2*M*HIDDEN*BPE + M*4, gpu) # read h, write x, per-row stat
    f = estimate_fused_gemm("router+res+rms_prologue", M, EXPERTS, HIDDEN, 1,
                            Epilogue(a_factor=2.0,                    # residual second-operand read
                                     aux_smem_per_tile=lambda m0,n0: m0*4), gpu)  # rmsnorm K-reduction
    return (g.time_s + r.time_s + n.time_s)*1e3, f.time_s*1e3
```

**Model-labeling note for the writeup:** decompose the b1 fused est into (residual → `a_factor`) + (rmsnorm → `m0*4` aux) and confirm the A-read is counted **once** for the norm (no double-count). For any *pure-RMSNorm, no-residual* prologue variant, use `a_factor=1.0 + aux m0*4` (repo F3) so the RMSNorm cost is never conflated with the residual read. Standalone-baseline rmsnorm traffic is `2·M·HIDDEN·BPE + M·4` (physically must write normalized `x`), deviating from the estimator's `RMSNORM_TRAFFIC` constant (`M·HIDDEN·BPE + M·4`, which assumes the norm output is itself fused forward — the F3 assumption); state this deviation. Emit stock + `--t2-json` adjusted.

### B.6 Expected outcome (all figures estimator-predicted, not measured)

`measured_gain_verified` candidates: hand triton (both variants, no-split-K); forced template only if its evidence shows a single `triton_tem_fused_*` with no separate add/reduction (expected b2, not b1). Report geomean over drift-clean configs only.
- **b2:** **predicted** to fuse on working paths with a small positive gain on drift-clean configs (folds `3·M·HIDDEN` vector traffic net ~2×`M·HIDDEN` saved; tiny `n=256` makes that a large fraction of the standalone microbenchmark → a few % predicted, larger at large `M`, noisier at small decode `M`). Realized by forced template (if prologue-fusion fires) and by the hand kernel at ~vendor-GEMM speed.
- **b1:** no stock path fully fuses (SM gate + un-synthesizable K-reduction); only the hand dual-accumulator captures it, single-pass so **predicted** ~vendor-GEMM speed; est predicts a somewhat larger gain than b2. Reinforces the verdict — **residual prologue is (template/hand) fusable; RMSNorm-into-prologue needs a custom kernel** — but the absolute router-level payoff is small (tiny N/output, minor cost center; shrunk further by shared `x`/`h`).

### B.7 Attempt-then-drop

`addmm` drop up front (record reason). b2 `forced`: if evidence shows a separate add kernel remaining, document "stock/forced template did not fold the input add on this torch build" and fall back to the hand kernel as verified-fused. b1: expect residual-fold-at-best; rely on hand kernel for `measured_gain_verified`. Small decode `M ∈ {512,1024}` may be latency/occupancy-bound (few row-tiles on 24 SM) — flag mutually-inconsistent `gemm_only`/`unfused` + drift-far-from-1 rows and exclude from aggregate (prefer `BM=64` to raise tile count). Prefill `M=131072`: b1 unfused holds `attn,res,h,x` (~6 GiB + workspaces) — if it OOMs, drop the top prefill point to `65536` and document; the fused path uses fewer buffers.

---

## T6.C — top-k (k=8) selection as the router GEMM epilogue (attempt-and-DROP)

`measure_router_topk(...)` + `router_topk_configs(smoke)` in `rtx4060_fusion_measure.py`; rows into the T4 JSON schema with `"kind": "router_topk"`. **User explicitly doubts this fuses — this test is a rigorous feasibility analysis + one minimal attempt, and is a documented DROP** (C.6).

### C.1 The fusion + dims

`logits = x @ Wr` (`M=tokens`, `N=EXPERTS=256`, `K=HIDDEN=6144`) then `vals, idx = torch.topk(logits, 8, dim=-1)`. Router is DENSE → `M=tokens` directly. `Wr=[6144,256]` = 3 MB. DECODE `tokens ∈ {512..16384}`; PREFILL `tokens ∈ {8192,32768,131072}` (all fit). Softmax over the 8 gates is NOT part of the fusion.

### C.2 Explicit unfused baseline

`best_unfused_ms = min(` **eager:** cuBLAS `x@Wr` then separate `torch.topk`; **custom-row-topk baseline:** same cuBLAS GEMM then a standalone row-resident Triton top-k reading `logits` once (so the baseline is not penalized by torch.topk's generic sort) `)`.

### C.3 6-path verdict — 5 of 6 cannot fuse (still run as recorded evidence)

`unfused` baseline; `addmm` **no** (β-accumulate is residual-only, no selection epilogue); `compiled`/`nocg` **no** (`torch.topk` lowers to `cub::DeviceSelect`/radix-select — an Inductor fusion barrier, stays a separate kernel); `forced` **no** (emits a template GEMM but top-k is not a pointwise epilogue → separate topk kernel survives — the recorded structural proof); `triton` **only via BN=256** (C.4). Two structural reasons: (1) **cross-tile** — top-k over N=256 is a full-row selection; any `BN<256` tiling spreads the row across CTAs so no tile-local epilogue can produce the global top-8; (2) **not elementwise** — selection/partial-sort is outside the cuBLASLt/CUTLASS/Inductor epilogue vocabulary (pointwise/broadcast/simple-row-reduce only). Run compiled/nocg/forced anyway (cheap) to record via `profile_kernels` that a distinct `topk`/`sort`/`DeviceSelect` kernel survives → `fused_verified=False` by construction; that evidence IS the deliverable for those paths.

### C.4 The one viable path: custom Triton, BN=N=256

Because `EXPERTS=256` is small, force `BN=256` (no N-tiling) → one CTA computes the whole `logits[BM,256]` row-block, top-k = within-CTA 8× iterative arg-max-and-mask over the 256-wide fp32 accumulator, writing `vals[BM,8]` (+ `idx[BM,8]`); grid `(cdiv(M,BM),)`. Autotune `BM ∈ {16,32,64}`, `BK ∈ {32,64}`, warps {4,8}, stages {1,2} (mirror `TRITON_CANDS`/`tune_triton_swiglu`, catch compile/launch exceptions, reject non-fitting tiles). **Feasible on the 4060** (99 KiB SMEM): `BM≤32` fits comfortably (fp32 `[BM,256]` accum ≤ 32 KiB + ~36 KiB stream, may stage to SMEM); `BM=64` needs a register-resident accumulator (128 accum regs/thread @ 4 warps → spills). The open question is speed, not possibility. Numerics: compare `(vals, idx)` to `torch.topk(logits.float(), 8)` on values (sorted top-8 allclose, bf16 rtol≈2e-2) and index SET (not order; bf16 ties can pick different equal-valued experts — acceptable, routing is tie-insensitive post-softmax).

### C.5 Estimator side (est-vs-measured cell — flag as structure-blind traffic-only bound)

The estimator has **no top-k primitive** and cannot model the selection compute or the BN=256 register/occupancy tax, so any number here is a **memory-traffic-only bound, NOT a prediction** — tabulate the cell as such and exclude from delivered-fraction stats (T6.0.4 guard). Since this test is a documented DROP, the exact Epilogue parameters below only feed that flagged est-vs-measured cell:

```
unf = estimate_gemm_grouped("router", M, 256, 6144, 1, gpu).time_s \
    + estimate_vector_kernel("topk_select", M*256*BPE + M*8*6, gpu).time_s
fus = estimate_fused_gemm("router+topk", M, 256, 6144, 1,
        Epilogue(out_factor=(8*(BPE+4))/(256*BPE),          # ≈0.094: write 8 vals + 8 int32 idx,
                                                            #   not the 256 logits
                 extra_hbm_once=M*8*4,                       # index-write extra HBM (8 int32/row)
                 aux_smem_per_tile=lambda m0,n0: m0*256*4),  # fp32 [m0,256] row accumulator
        gpu).time_s
```
(`out_factor` writes 8 vals + 8 indices instead of 256 logits; `extra_hbm_once` charges the index write; `aux_smem` the full-row fp32 accumulator.) It will **predict** a small memory-only win (~1.03–1.04×) the hardware need not deliver — that mismatch is the finding.

### C.6 Expected outcome + DROP protocol (all figures estimator-predicted)

The only payoff is removing the `logits` HBM round-trip = `2·M·256·BPE`, a fixed **~3.5%** of the router GEMM at every M (router compute-bound at N=256,K=6144: `2/170e9 ÷ 6144/18.4e12 = 0.035`), while standalone top-k is already ~1.75% of the router (**predicted** ≈6 µs at tokens=2048, ≈395 µs at 131072). To win, the BN=256 custom kernel must stay within ~3.5% of cuBLAS — but a 256-wide fp32 accumulator with no N-parallelism and low occupancy is exactly what a tuned library avoids on this thin-N shape → **predicted `measured_gain ≤ 1.0`, neutral-to-negative.**

**DROP condition:** if across the drift-clean sweep (a) no stock path fuses (profiler shows a surviving separate topk/sort — expected) AND (b) the custom BN=256 kernel's `measured_gain ≤ 1.0`, REMOVE Test C from the results table and record: *"top-k is a cross-tile selection/sort, not a tile-local elementwise epilogue; stock fusers keep it as a fusion-barrier kernel (verified), and the single viable custom route (BN=256 full-row-resident, feasible on 4060 SMEM/registers) pays a GEMM-efficiency tax exceeding the fixed ~3.5% logit-round-trip ceiling, so it does not beat GEMM+torch.topk. Estimator predicted a small memory-only win it cannot represent. Dropped."* Keep the raw JSON + profiler evidence so the drop is auditable. **KEEP condition:** if some drift-clean config shows `measured_gain > 1.0` with the custom kernel at ~vendor-GEMM speed, report it as the rare win with its exact tile and regime — do not drop then.

---

## T6.D — MoE FFN, single-expert, three fusion levels (up_gate → SwiGLU → down, incl. on-chip F6)

Reuse `measure_swiglu` for L2; add the F6 kernel + a **tpe-parametrized F6 estimator** (D.5, HARD prerequisite). Rows → `rtx4060_fusion.json`. Closest hardware test of the real GLM MoE FFN fusion. Three levels: **L1** unfused (up_gate GEMM → separate SwiGLU → down GEMM, 3 kernels); **L2** SwiGLU folded into up_gate epilogue (half-width write, `out_factor=0.5`) + vendor down (2 kernels); **L3/F6** up_gate→SwiGLU→down as ONE kernel, activated intermediate resident in SMEM (1 kernel). Single-expert modeling per T6.0.2 (run one expert at `M=tpe`, layer ≈ 256×; caveats a/b/c apply).

### D.1 Exact dims (per expert, `tpe = tokens/32`)

| stage | op | `[M,N,K]` | weight (bf16) |
|---|---|---|---|
| up_gate | `gu = x @ Wug` | `[tpe, 4096, 6144]` | `Wug[6144,4096]` = 50.3 MB |
| SwiGLU | `act = silu(gu[:,:2048]) * gu[:,2048:]` | `[tpe,4096]→[tpe,2048]` | — |
| down | `out = act @ Wdn` | `[tpe, 6144, 2048]` | `Wdn[2048,6144]` = 25.2 MB |

`Wug+Wdn = 75.5 MB/expert`. Prefill `tpe=4096` single-expert footprint ~0.17 GB; grouped-8 ~1.8 GB. All ≤ 8 GB. Sweep = T6.0.3 (`tokens=8192→tpe=256` kept in both regimes).

### D.2 Explicit unfused baseline (L1)

`FFN_L1 = t(up_gate GEMM) + t(SwiGLU vector) + t(down GEMM)`, each a separate eager kernel: `gu = x @ Wug`; `act = F.silu(gu[:,:2048]) * gu[:,2048:]`; `out = act @ Wdn`. Record the best unfused realization: **`best_unfused = min(eager up_gate + separate SwiGLU, nocg)` for the up_gate+SwiGLU part, plus the common vendor `down` GEMM** (`nocg` = max-autotune-no-cudagraphs of `swiglu_ffn` = vendor GEMM + one fused pointwise kernel = the estimator's unfused model).

### D.3 6-path verdict

**L2 = tile-local epilogue (fusible):** each output tile over the INTER half-width needs gate cols `[n:n+BN]` and up cols `[n+2048:...]` from the same x row-block over K=6144 into two register accumulators; `silu(g)*u` is elementwise on those → folds cleanly. `unfused` baseline; `addmm` N/A (no residual); `compiled` partial (24-SM gate → vendor GEMM + separate fused pointwise epilogue kernel, records cudagraph copy overhead); `nocg` = **best UNFUSED**; `forced` **FUSES** (patches `is_big_gpu→True` + TRITON backend → Triton GEMM template with folded SwiGLU epilogue; `fused_verified_forced` must confirm no separate silu/mul kernel); `triton` **FUSES by construction** (existing `triton_swiglu` dual-accumulator, half-width write; reuse verbatim at hidden=6144, inter=2048). → **Run `measure_swiglu(name, M=tpe, hidden=6144, inter=2048, count=1, batched=False)`**; `L2_up_swiglu_ms = its best_fused_ms`.

**L3 = cross-tile reduction (only a hand kernel fuses):** `down` contracts over `K=INTERMEDIATE=2048` = the SwiGLU output width, so a down output tile needs the ENTIRE activated row → ALL up_gate tiles for those rows must finish + reduce through SwiGLU first. `unfused/addmm/compiled/nocg/forced` **cannot fuse L3** (none fuses two GEMMs across a shared contraction; forced only folds a pointwise epilogue). `triton` (NEW hand kernel) = only feasible path (D.4).

### D.4 The F6 (L3) custom kernel — feasible at these dims

**SMEM (99 KiB = 101376 B):** binding constraint = resident `act[m0, 2048]` bf16 = `m0·4096 B`, + down-output accum (`m0·128·2`) + ~16 KiB k-slice streams:

| `m0` | activated | + accum + 16 KiB | fits 99 KiB? |
|---|---|---|---|
| **16** | 64 KiB | **84 KiB** | **yes** |
| 32 | 128 KiB | 152 KiB | no |

→ **`m0` hard-capped at 16** (MMA-min BM, only power-of-two divisor of every `tpe` that fits). Forces `mt = tpe/16` row-blocks; both weights (75.5 MB > 32 MB L2) exceed L2 → each row-block re-reads Wug+Wdn → **total weight traffic = `mt·75.5 MB`** vs `1·75.5 MB` unfused. That `mt×` re-read is the entire F6 penalty and the thing to measure. The 64 KiB resident leaves ~35 KiB → `num_stages ≲ 2` (a second, smaller penalty). Kernel shape: grid `(mt,)`, `BM=16`, `act_smem: bf16[16,2048]` persists phase1→phase2; phase1 fills `act_smem` (up_gate + SwiGLU), phase2 reads it (down, Wdn streamed once). Autotune `(BN_ug,BK_ug,BN_d,BK_d,warps,stages)` (start stages=2, warps∈{4,8}). Numerics vs fp32 `out32 = (silu(gu32[:,:2048])*gu32[:,2048:]) @ Wdn32`, tol `rel_max ≤ max(2·eager_rel,5e-2)`.

**ATTEMPT (feasible, m0=16 fits); DROP-and-document only if** (i) the kernel cannot keep `act_smem` in SMEM at BM=16 (spill/compile fail) → fall back to estimator-only F6, or (ii) numerics fail even at fp32 activated (needs m0<16 → not MMA-legal → confirms infeasible). Record the reason + the m0 SMEM table; report L3 as estimator-only.

### D.5 Estimator hooks — tpe-parametrized F6 is a HARD PREREQUISITE (not a risk)

L1/L2 already covered (`estimate_gemm_grouped`, `estimate_vector_kernel("activation", 3·tpe·2048·2)`, `est_swiglu_ms` — all m/n/k-parametrized). **F6 is NOT:** the repo's `estimate_ffn_fused` reads `m = TOKENS_PER_EXPERT` as a module global fixed at 64 (`fusion_time_estimator.py:211`; `TOKENS_PER_EXPERT = BATCH*TOP_K//EXPERTS = 64`). Run unmodified, **every F6 est is computed at m=64 regardless of the swept `tpe`, so the entire tpe-crossover story (the test's headline finding) cannot appear.** Therefore, as a **required spec step (do this before running the sweep)**, add a tpe-parametrized variant that takes `m` as an argument:

```python
def estimate_ffn_fused_m(m, count, gpu):   # parametrize estimate_ffn_fused by m (do NOT read the global)
    w_ug = HIDDEN*(2*INTERMEDIATE)*BPE; w_dn = INTERMEDIATE*HIDDEN*BPE; x_out = 2*m*HIDDEN*BPE
    ops = (2*m*(2*INTERMEDIATE)*HIDDEN) + (2*m*HIDDEN*INTERMEDIATE); STREAM = 16*1024; best = None
    for m0 in divisors(m):
        if m0 < MMA_MIN_BM: continue                     # reproduces the m0=16 MMA/SMEM cap
        mt = m//m0; resident = m0*INTERMEDIATE*BPE + m0*128*BPE + STREAM
        if resident > gpu.smem_per_block_bytes: continue
        traffic = count*(x_out + mt*(w_ug + w_dn))       # weights re-read mt x  <-- the crossover driver
        tiles   = count*mt*max(1, HIDDEN//128)
        t = _roofline(gpu, count*ops, traffic, tiles, resident, 2)[0]
        if best is None or t < best[0]: best = (t, m0, mt)
    return best                                          # None => infeasible
```
**Acceptance for the parametrization (assert before trusting any F6 est):** (1) at `tpe=64` it reproduces the known F6 = 0.259×; (2) `estimate_ffn_fused_m(16,…)` and `estimate_ffn_fused_m(512,…)` differ by the ~`mt` weight-reread factor (F6 time must scale with `mt`, not stay flat at m=64); (3) the enumerated `m0` is capped at 16 (SMEM). Pass `m0/mt` from the swept `tpe`, not the global.

**Predicted numbers (estimator, `rtx4060-measured` — the est side of est-vs-measured; NOT measurements):**

| `tpe` (tokens) | `mt` | isolated up+SwiGLU (L2) | full-FFN `r_L2` | full-FFN `r_L3` (F6) | F6 per-expert / ×256 layer |
|---|---|---|---|---|---|
| 16 (512) | 1 | 1.005× | 1.003× | **1.005× (slight WIN)** | 0.446 ms / 114.3 ms |
| 32 (1024) | 2 | 1.010× | 1.007× | 0.508× | — |
| 64 (2048) | 4 | 1.020× | 1.014× | **0.259×** | 1.786 ms / 457 ms |
| 128 (4096) | 8 | 1.024× | 1.016× | 0.162× | — |
| 256 (8192) | 16 | 1.026× | 1.017× | 0.152× | — |
| 512 (16384) | 32 | 1.026× | 1.017× | 0.152× | 14.29 ms / 3657 ms |
| 1024 (32768) | 64 | 1.026× | 1.018× | 0.150× | — |
| 4096 (131072) | 256 | 1.032× | 1.022× | 0.150× | 114.3 ms / 29256 ms |

### D.6 What to measure + expected outcome (figures predicted, not measured)

Per `tpe`, single expert (T6.0.4 methodology): **L1** `FFN_L1_ms` (+ best_unfused via nocg); **L2** `FFN_L2_ms = L2_up_swiglu_ms(best verified) + down_gemm_ms`; **L3** `FFN_L3_ms = F6_kernel_ms` (verified by construction, numerics OK). Ratios `r_L2 = L1/L2`, `r_L3 = L1/L3`, isolated `r_up_swiglu`. Layer `256·FFN_*_ms` (note merge excluded, caveat b). **F6 headline — weight-re-read penalty:** tabulate `FFN_L3/FFN_L1` vs `mt`: (a) direct — `mt=1` at `tpe=16` is the no-re-read best case, per-token F6 grows ≈`mt×` in the weight-bound region; (b) traffic — if `ncu dram__bytes` available confirm F6 weight bytes ≈ `mt·75.5 MB`, else compute `mt` analytically.

**Expected (predicted):** **L2 fuses and modestly wins** (isolated 1.005×→1.032× growing with tpe; diluted to `r_L2` 1.003×→1.022× because the common down GEMM does not shrink) — the realizable positive fusion, hand `triton`/`forced` at ~vendor-GEMM speed (unlike the C500 3.4× tax). **L3/F6 has a sharp crossover at `mt = 1↔2` (tpe 16↔32):** parity/slight win at tpe=16, then a cliff, flooring at ≈0.15× (~6.7× slower) for `tpe≥256` — the `mt×` weight-re-read signature confirming the F6=SKIP verdict at deployment batch sizes. Expect measured F6 **≤** estimator at small tpe (the estimator's `tiles=mt·48` occupancy term is optimistic; a faithful `grid=(mt,)` kernel launches only `mt` CTAs, under-filling 24 SMs); the grouped cross-check (D.7) restores occupancy and is the fairer F6 test.

### D.7 Grouped-bmm (8-expert) cross-check

At a decode `tpe=64` and a prefill `tpe=512`: **L1/L2** `measure_swiglu(..., count=8, batched=True, do_triton=False)` (grouped bmm for up_gate+SwiGLU, vendor bmm for down; hand kernel is dense-only so only compiled/forced verify the fold in grouped mode — fine, this targets GEMM timings), check `grouped_ms ≈ 8×single_expert_ms`. **F6** launch with `grid=(8·mt,)` (row-block `e·mt+b`, weights indexed by expert `e`) — restores realistic occupancy (8·mt CTAs); report grouped-F6/grouped-L1 alongside the single-expert ratio (the `mt×` traffic penalty is identical, occupancy is the only change).

---

## T6.E — residual2 into the expert-MERGE epilogue (`r2f`, memory-bound)

`measure_merge_r2f(...)` in `rtx4060_fusion_measure.py`; wire into `main()`. Unlike A/B/D (GEMM + epilogue), the merge is a **cross-expert reduction with no matmul** — pure memory-bound, the highest-ceiling, cleanest fusion in the study.

### E.1 The fusion + dims (synthetic dense merge, per T6.0.2b)

After per-expert `down`, each token has TOP_K=8 outputs `[HIDDEN]`; combine weight-sums them with the gates and adds `residual2`:
`out[t,:] = residual2[t,:] + Σ_{e=0..7} gate[t,e]·expert_out[t,e,:]` → `[tokens, HIDDEN]`. Cross-expert, so modeled as a SYNTHETIC dense merge (not one expert):

| tensor | shape | dtype |
|---|---|---|
| `expert_outs` | `[tokens, 8, 6144]` | bf16 |
| `gates` | `[tokens, 8]` | bf16 |
| `residual2` | `[tokens, 6144]` | bf16 |
| `out` | `[tokens, 6144]` | bf16 |

`M = tokens` (dense over ALL tokens). Prefer the stacked `[tokens,8,6144]` layout (contiguous last dim → coalesced `expert_outs[t,e,h0:h0+BN]` reads, near-peak BW).

### E.2 Explicit unfused baseline

Two kernels, `[tokens,HIDDEN]` `merged` written+read from HBM: `merged = (expert_outs*gates.unsqueeze(-1)).sum(dim=1)` (read `8·T·H` + write `T·H`) then `out = merged + residual2` (read `2·T·H` + write `T·H`). **`best_unfused_ms = min(`** eager two-op (C500-comparable) **,** clean unfused = `torch.compile` of the *merge-only* expr (inductor fuses mul+sum into ONE reduction ≈ the estimator's 9·T·H model) then a separate eager `+ residual2` **)** (avoids penalizing the baseline with eager's mul-temporary traffic; the fair 2-kernel reference).

### E.3 6-path verdict

`unfused` baseline (merge reduction + separate residual-add). `compiled` **FUSES** (`residual2` add is tile-local: `out[t,h]` uses `residual2[t,h]` at the same index as the reduction result — no cross-tile reduction; contrast F4 SwiGLU's disjoint column-slices; inductor folds mul+sum+add into one `triton_red_fused_*`; track `compiled_cudagraph_input_copy_us` — can dominate this cheap op). `nocg` **FUSES; expected best** (same fused reduction, no cudagraph copies). `triton` optional (fuses by construction, memory-peak upper bound — see E.4; NOT required to capture the win → `needs_custom_kernel = False`, the exact case the C500 could not and the contrast to SwiGLU which needed a hand kernel). `addmm` **N/A** (no GEMM; only analog `torch.baddbmm(residual2[:,None,:], gates[:,None,:], expert_outs)` = degenerate `m=1,k=8,n=6144` batched GEMM → tensor cores idle, memory-bound + reshape overhead — attempt as a curiosity, drop with a note if it errors or runs >1.5× the fused reduction). `forced` **N/A — DROP** (no mm/bmm node to template; op lowers to a reduction → same kernel as nocg; record `"forced": "N/A: no GEMM template (op is a reduction)"`).

### E.4 Optional hand Triton kernel (upper bound)

```
# out-tile [BM tokens × BN hidden]; grid=(cdiv(T,BM), cdiv(H,BN))
acc = tl.zeros((BM,BN), tl.float32)
for e in range(8):                                          # TOP_K=8, unrolled
    g = tl.load(gates + rows*8 + e)                         # [BM]
    x = tl.load(expert_outs + rows[:,None]*8*H + e*H + cols[None,:])   # [BM,BN]
    acc += g[:,None]*x
acc += tl.load(residual2 + rows[:,None]*H + cols[None,:])              # tile-local residual2 add
tl.store(out + rows[:,None]*H + cols[None,:], acc.to(out.dtype.element_ty))
```
Tune `(BM,BN,num_warps)` as in `tune_triton_swiglu`; numerics vs fp32 ref, `max(2×eager_rel,5e-2)`.

### E.5 Estimator + token-independence assertion (fixes the missing-131072 gap)

```python
def est_merge_r2f_ms(T, H, gpu):
    unf_merge = estimate_vector_kernel("merge",     (8*T*H + T*H)     * BPE, gpu)  # 9·T·H
    unf_add   = estimate_vector_kernel("residual2", (3*T*H)           * BPE, gpu)  # 3·T·H
    fused     = estimate_vector_kernel("merge+res2",(8*T*H + T*H + T*H)*BPE, gpu)  # 10·T·H
    return (unf_merge.time_s + unf_add.time_s)*1e3, fused.time_s*1e3
```
Unfused traffic `12·T·H·BPE`, fused `10·T·H·BPE` → the fusion eliminates exactly one `[tokens,HIDDEN]` round-trip (`2·T·H·BPE`). **`estimated_gain = 12/10 = 1.20`, and it is PROFILE-INDEPENDENT and TOKEN-INDEPENDENT** — both sides are `traffic/bw` (bandwidth cancels → stock and T2-adjusted give the identical 1.20, report once), and the ratio has no T dependence (it is `12·T·H / 10·T·H`, T cancels). Gate/index reads (`T·8`) are `1/H` negligible.

**Token-independence requirement (mandatory, so the un-measurable 131072 point is defensibly inferable rather than silently absent):** measure the **verified-fused gain at 512, 8192, and 32768** and **assert it is flat** across the measurable range (predicted ~1.12–1.18× delivered, i.e. within the 1.20 ideal minus a stable strided-reduction/gate-read fraction). Record this flatness explicitly in the results as `merge_gain_token_independent: true/false` with the three-point spread. Only if flat may the writeup state that the missing `131072` point is covered by token-independence; if NOT flat, flag it and do not claim 131072 coverage.

### E.6 What to measure + expected outcome (figures predicted, not measured)

Per config: `merge_only_ms` (bare merge reduction — also the drift-probe kernel, re-measured at config end, clean iff `|drift−1|≤0.05`), `unfused_ms`, clean-unfused, `fused_paths = {compiled_ms, compiled_nocg_ms, [triton_ms], [baddbmm_ms]}`, `best_unfused_ms`, `best_fused_ms`, `measured_gain`, `measured_gain_verified`, est trio, `fused_verified` + `kernel_evidence`, numerics, clocks. Fused-verification: adapt `vendor_fused_ok` with `act_markers=("add","residual","elementwise")` — fused iff one `triton_red`/`triton_poi` kernel absorbs the `residual2` read with no trailing standalone add; do NOT require `_is_gemm_template` (no GEMM). This op has **NO compute-bound crossover** (no matmul) → `estimated_gain` ~1.20 at every token count; the sweep confirms delivered-gain stability, not a regime change. **Expected:** compiled/nocg FUSE out of the box (verified: one kernel) and deliver a clear positive **predicted** gain ~+12…+18% (delivered < 1.20 from the strided 8-expert reduction not hitting full peak + unmodeled gate/index reads). Strongest verdict-A datapoint: a memory-bound fusion stock `torch.compile` captures with NO custom kernel, unambiguously bandwidth-bound at all sizes (ratio depends only on the memory clock — robust to the unlocked-clock caveat), predicting a gain well above noise.

### E.7 Token sweep + attempt-and-drop (COVERAGE FIX)

**Sweep (`M = tokens`):** decode `{512,1024,2048,4096,8192,16384}`; prefill `{32768}` **and `{49152}` (add it — it fits ~7.8 GB peak, see below)**. **`131072` is INFEASIBLE at 8 GB — drop with arithmetic:** `expert_outs[131072,8,6144]` bf16 alone `= 8·131072·6144·2 = 12.9 GB > 8 GB`. Per-token resident ≈ 144 KB (`expert_outs 96 KB + residual2 12 KB + out 12 KB + unfused merged 12 KB + transient fp32 ref 24 KB`) → `T ≲ 40k` is the practical ceiling for the full baseline. **`49152` (~7.8 GB peak) is the borderline largest MEASURABLE prefill point — include it, freeing the fp32 reference and the unfused `merged` intermediate before allocating so it fits.** The `131072` point is then covered by the E.5 token-independence assertion (gain flat 512→32768→49152 ⇒ the 1.20-class ratio is token-independent ⇒ the un-measurable 131072 gain is the same), NOT silently absent. `baddbmm`: attempt once, keep if ≤1.5× fused reduction else `baddbmm_note` + drop. `forced`: do not run, record N/A. Optional `scatter_add`/`index_add` cross-check to confirm the dense synthetic merge is representative of the ratio (not required).

---

## T6 Deliverables

- **New scripts:** `rtx4060_rmsnorm_fuse.py` (A), `rtx4060_router_prologue.py` (B). **New measure functions in `rtx4060_fusion_measure.py`:** `measure_router_topk` + `router_topk_configs` (C), the L2 reuse of `measure_swiglu` at GLM per-expert dims + the F6 hand kernel (D), `measure_merge_r2f` wired into `main()` (E). **New estimator helper in `fusion_time_estimator.py`:** `estimate_ffn_fused_m(m, count, gpu)` (D.5, HARD prerequisite — do NOT rely on the m=64 global).
- **Extend `rtx4060_fusion.json`** with the new configs: Test C rows (`"kind":"router_topk"`), Test D rows (`{tpe, tokens, regime, mt, ffn_L1_ms, ffn_L2_ms, ffn_L3_ms, up_swiglu_best_fused_ms, down_gemm_ms, r_L2, r_L3, r_up_swiglu, est_ffn_L1_ms, est_ffn_L2_ms, est_ffn_L3_ms, est_r_L2, est_r_L3, f6_fused_verified, f6_numerics_ok, f6_m0, f6_mt, forced_over_gemm_ratio, gemm_drift_ratio, layer_L1_ms, layer_L2_ms, layer_L3_ms, f6_kernel_evidence, [f6_dram_bytes], clocks}`), and Test E rows (`{tokens, regime, merge_only_ms, unfused_ms, clean_unfused_ms, compiled_ms, compiled_nocg_ms, [triton_ms], [baddbmm_ms], best_unfused_ms, best_fused_ms, measured_gain, measured_gain_verified, est_unfused_ms, est_fused_ms, estimated_gain, merge_gain_token_independent, fused_verified, kernel_evidence, numerics, drift, clocks}`). Tests A/B write their own JSONs (`rtx4060_rmsnorm.json`, `rtx4060_router_prologue.json`) with the full per-path schema of T6.0.4.
- **Append five sub-sections to `rtx4060_sim_real_results.md`** (one per test): each with its est-vs-measured table (drift-clean geomean), which paths fused (kernel evidence), and the verdict. **Reporting guards in the tables:** (A) mark the `mla_o` RMSNorm-epilogue estimator cell `est=INVALID (structure-blind)` and exclude it from delivered-fraction stats; (C) mark the top-k est cell as a traffic-only bound (not a prediction), exclude from stats; every quoted timing/percentage in the write-ups is tagged `est`/`predicted` unless a cell is an actual measurement; (E) record `merge_gain_token_independent` + the 512/8192/32768(/49152) spread so the dropped 131072 point is defensibly inferable.
- **Log all steps in `notes/rtx4060_worklog.md`** per the T4 convention.