# RTX 4060 sim–real fusion gap — worklog

Task: `RTX4060_SIM_REAL_TASK.md`. Date: 2026-07-20. Agent: Claude (ultracode mode).

## T1 — Environment + device discovery (2026-07-20)

- `nvidia-smi`: NVIDIA GeForce RTX 4060 Laptop GPU, driver 610.53, CUDA UMD 13.3, 8188 MiB VRAM.
  Idle: P8, 5 W, 58 °C, 553 MiB in use (GPU also drives the WSL2 display — noted as a noise source).
- Env: python 3.13.12 (miniconda), torch 2.11.0+cu130 (CUDA 13.0), triton 3.6.0. ≥py3.11 requirement met.
- `torch.cuda.get_device_properties(0)`:
  - name: NVIDIA GeForce RTX 4060 Laptop GPU, cc 8.9 (Ada)
  - SM count: **24** (matches profile)
  - total_memory: **8.00 GiB** (confirmed 8 GB)
  - L2: **32 MiB**
  - SMEM/block: 48 KiB default, **99 KiB opt-in** (matches profile's 99 KiB), 100 KiB/SM
  - regs/SM: 65536
- Clocks: idle 210 MHz gr / 405 MHz mem; max 3105 MHz gr / 8001 MHz mem. Power limit reports N/A
  (nvidia-smi header shows 35 W cap — laptop part).
- **Clock locking NOT possible**: `nvidia-smi -lgc/-lmc` denied (needs root; sudo requires password;
  WSL2). Per task §7 fallback: record clocks during each measurement run instead. Profile was
  calibrated at locked 1500/5501 MHz → expect unlocked boost to run faster; will adjust profile via
  `dataclasses.replace` in T2 if measured peaks differ (task §T2 anticipates this). Relative
  fused/unfused results are robust to this (task §7).

## Prep — repo digest + script authoring (2026-07-20/21)

- Wrote `rtx4060_common.py`: shared timing module mirroring `metax_glm_measure.py` methodology
  (per-iter cuda Events, median) tightened to task §7 (warmup ≥15, iters ≥30), plus a background
  `ClockSampler` (nvidia-smi poll) since clocks can't be locked.
- Verified estimator imports work in this env: `GPUS['rtx4060-measured']` = 24 SM, 96 TC ×
  128 flop/clk × 1.5 GHz = 18.4 TF/s, 170 GB/s, smem/block 101376 B, L2 32 MiB.
- Read `fusion_time_estimator.py` first-hand. F4 fused = GEMM with `out_factor=0.5` epilogue
  (writes activated half-width), unfused = GEMM + vector kernel `M·3·inter·BPE`. F1 fused = GEMM +
  `extra_hbm_once=M·N·BPE` + residual SMEM tile; unfused = GEMM + `3·M·N·BPE` vector kernel.
  All estimator times in seconds.
- NOTE: task §3-T4 writes mla_o as `[M,6144]@[6144,16384]` (n=16384) but the GLM/C500 layer used
  `[M,16384]@[16384,6144]` (n=6144, residual on hidden). Decision: measure BOTH orderings for F1
  (cheap), estimator dims matched to each.
- Launched ultracode prep workflow `wf_fefce3fb-e32`: 5 parallel digest readers → 3 script authors
  (`rtx4060_peak.py` T2, `rtx4060_gemm_validate.py` T3, `rtx4060_fusion_measure.py` T4, smoke-tested
  on tiny dims only) → per-script adversarial verifier checking units/dims/estimator-replication
  against actual sources. Full GPU sweeps deliberately NOT run by agents (serialized later).

## Prep findings (pre-run, from authoring/smoke) — 2026-07-21

1. **Inductor refuses Triton GEMM templates on 24-SM GPUs**: `torch/_inductor/utils.py
   is_big_gpu()` hard-requires ≥68 SMs; on the 4060 max-autotune logs "Not enough SMs to use
   max_autotune_gemm mode" and keeps the vendor GEMM + a separate Triton pointwise epilogue
   kernel. Fix applied in `rtx4060_fusion_measure.py`: a `force_triton_templates` context
   (patch `is_big_gpu` + `max_autotune_gemm_backends="TRITON"`) so the task's key path — a
   Triton GEMM template with fused epilogue — is actually emitted and measured. Distinct code
   objects per compile variant (dynamo caches on the code object; inductor config is not in
   its key).
2. **Cudagraph static-input-copy contaminates timed compiled calls** (~2–3 % at full dims,
   same order as the effect under test) → added a `max-autotune-no-cudagraphs` variant per
   config; the cudagraph copy time is also tracked separately.
3. **STRUCTURAL: SwiGLU cannot fold into the up_gate GEMM epilogue via inductor** even when
   templates are forced: smoke evidence shows `triton_tem_fused_mm_0` + separate
   `triton_poi_fused_mul_silu_slice_1`. Reason: silu(gu[:, :inter])*gu[:, inter:] combines
   TWO disjoint column-slices of the GEMM output — different output tiles — while template
   epilogue fusion is elementwise-on-own-tile only. A genuinely fused SwiGLU needs a
   dual-accumulator kernel (gate and up tiles computed together). We wrote one (hand Triton,
   mini-autotuned over 6 tile configs); it is the ONLY truly-fused SwiGLU path on this stack.
   (cuBLASLt has no SwiGLU epilogue in torch's binding either.)
4. **Residual DOES fold**: forced path emits a single `triton_tem_fused_addmm_0` (verified,
   1 kernel); vendor `torch.addmm` = cutlass GEMM + Memcpy DtoD (beta-accumulate copy of res).
5. metax_fused_triton.py turned out to contain NO hand Triton kernel (its "fusion" was
   torch.compile) — the C500 "3.4× fusion tax" datapoint came from compile, not a hand kernel.
6. Added residual GLM ordering configs (n=6144, k=16384) alongside the task's (n=16384,
   k=6144) — the C500 measured the former; measuring both.
7. `measured_gain_verified` (best unfused realization / best VERIFIED-fused path) is the
   verdict metric; `measured_gain` (eager unfused / best "fused" label) kept for C500
   comparability. T3 author 522'd on first workflow run; resumed (cache-hit for the rest).

## T2 — peak specs (2026-07-21)

- First run (no clock warming): peak 19.52 TF/s (1.059×), bw 242.9 GB/s (1.429×) — clocks
  boosting above the locked-calibration point (gr spikes to 2070 MHz, mem to 7501 vs locked
  5501). Also exposed the **DVFS cold-start artifact**: short measurements after a cooldown ran
  at idle 210 MHz (2048×1024×1024 measured 2.2 TF/s).
- **Methodology fix**: `rtx4060_common.warm_clocks()` — ~200 ms continuous busy-GEMM (queued
  batches, no per-iter sync: idle micro-gaps prevent DVFS commit) immediately before every timed
  region, now called inside `med_time`. Reran T2+T3 with it.
- Final T2 (warmed): **peak 18.79 TF/s = 1.019× profile** (implied sustained tensor clock
  1529 MHz vs locked 1500 — essentially at the calibration point), **bw 214.8 GB/s = 1.264×**
  (unlocked mem clock ~7 GHz vs locked 5501). Adjusted profile emitted (clock_hz=1.529e9,
  bw=2.148e11) → used as `--t2-json` downstream. Files: `rtx4060_peak.json`.
- Note: adjusted profile shifts ridge OI from 108 → ~87 FLOP/B → estimator's predicted fusion
  gains shrink under adjustment (vector traffic relatively cheaper). Both variants recorded.

## T3 — single-GEMM estimator validation (2026-07-21)

- 24 shapes (squares 1024–6144, 5 tall/wide, all T4 FFN-stage dims), bf16, iters=40 warmup=15,
  clock-warmed. Files: `rtx4060_gemm_validate.json`, merged into `rtx4060_measured.json`.
- **Stock profile: geomean est/meas = 0.937, 96% within 1.5×, min 0.327 max 1.148 → IN the
  prior 0.72–0.96 band.** FFN subset: 0.915, 93% within 1.5×. Adjusted profile: 0.919 / 96%.
- Estimator is mildly optimistic on small shapes (1024³ ratio 0.69) and mildly pessimistic on
  the biggest (8192×16384×6144 ratio 1.148); T4's regime (up_gate/mla_o shapes) sits at
  0.88–1.08 → **trust T4's est-vs-meas comparison**.
- Known flaky datapoint: 2048×1024×1024 measured 0.73 ms (5.8 TF/s) in-sweep but 0.32 ms
  (13.4 TF/s) standalone with the same cuBLAS kernel — residual WSL2 DVFS noise; that shape is
  not among T4's measured GEMMs (T4's up_gate at h=1024 is 2048×2048×1024, which reads 0.88).
- Progression of the band as methodology improved: 0.768 (cold) → 0.885 (warmed, per-iter-sync
  warmer) → 0.937 (continuous warmer). DVFS, not the model, was the dominant error source.

## T4 — launched (2026-07-21)

- `python rtx4060_fusion_measure.py --out rtx4060_fusion.json --t2-json rtx4060_peak.json --moe`
  running in background (log: scratchpad/t4_full_run.log). 6 dense swiglu configs, 4 residual
  (task + GLM orderings), 2 MoE grouped-bmm; per config: eager unfused, gemm-only, max-autotune
  (cudagraphs), max-autotune-no-cudagraphs (best unfused realization), forced-Triton-template
  (is_big_gpu patched, TRITON-only backends), hand Triton dual-accumulator SwiGLU (mini-autotuned),
  addmm for residual; profiler kernel evidence + numerics per path.

## T4 take 2 — mid-run fix (2026-07-21)

- Killed the first full run at config 3/12: the hand-Triton kernel was failing the naive
  `allclose(triton, eager_bf16)` numerics check at h≥2048 and its timing was therefore dropped.
  Root cause: the hand kernel keeps gate/up in **fp32 accumulators** through the silu — MORE
  accurate than the eager reference, which rounds `gu` to bf16 before silu; at k≥2048 the
  eager path's own bf16 rounding exceeds rtol=2e-2 in the tails. Fix: judge every swiglu path
  against an **fp32 reference** (`ok iff rel_max ≤ max(2× eager's own rel_max, 5e-2)`), record
  timing regardless, gate the verdict metric on the flag.
- Also added a per-config drift probe (`gemm_only` re-measured at config end → `gemm_drift_ratio`)
  after observing `unfused < gemm_only` on the flaky 2048×2048×1024 shape (impossible except as
  DVFS drift).
- Early real datapoints from take 1 (h=2048/4096, valid rows): fusion paths all SLOWER than
  unfused (meas_gain ≈ 0.90–0.96 vs est_gain 1.04–1.08); forced Triton template 1.24–1.30× the
  vendor GEMM (a real but modest fusion tax, nothing like C500's 3.4×); cudagraph copy visible
  in compiled_ms. Full verdict awaits take 2.

## T4 — results (2026-07-21)

- Full run (12 configs: 6 dense swiglu, 4 residual incl. GLM ordering, 2 MoE grouped-bmm)
  completed; `rtx4060_fusion.json`. Headline: verified-fused gain geomean **1.0368 on
  drift-clean configs vs estimator 1.0461 (adjusted) / 1.0569 (stock)** — the predicted gain
  materializes on genuinely-fused kernels running at ~vendor speed (hand Triton SwiGLU
  0.94–0.99× gemm-only; forced Triton addmm template 1.00×).
- Default torch.compile max-autotune fused 0/12 (SM-count gate + structural SwiGLU limit +
  cudagraph input-copy tax makes it net-negative vs eager). addmm beat mm+add on all 4 residual
  configs (+0.6–1.2%) — the C500's "addmm 1% slower" does NOT reproduce on CUDA.
- Thermal reality of the 35 W part: sustained clocks sag 1725→1140 MHz across the run; drift
  probe flagged 4/12 configs (ratio 1.09–1.35). The 2 throttled M8192 swiglu configs show
  negative gains WITH a degraded fused kernel (1.29–1.35× gemm) → classed inconclusive per task
  §5 criterion 3, not counted against the verdict.
- Numerics: every fused path ≤1 bf16 ulp from eager (verified elementwise: 99.8% within 1 ulp;
  addmm/hand-kernel are MORE accurate than eager vs fp32 truth — single vs double rounding).
  The JSON's strict allclose flags on residual rows are annotated false alarms.

## T5 — verdict (2026-07-21)

**A — fusion algorithm sound; C500 null was tooling.** Written up in
`rtx4060_sim_real_results.md` (est-vs-meas tables, fused-path inventory, §5-criteria mapping,
caveats). Sharpener: even on NVIDIA the gain is only collectable via torch.addmm (F1) or a
custom dual-accumulator kernel (F4); stock compile paths reproduce the C500 null (geomean 0.992
C500-convention). Grouped-MoE fused epilogues don't exist on either stack — the custom-kernel
recommendation for C500 carries an extra "grouped" requirement.

## Review round (2026-07-21)

- Ran a 3-auditor adversarial review workflow (numbers / logic / completeness) of
  `rtx4060_sim_real_results.md` against the raw JSONs and the task spec. Verdict-A direction
  survived; the writeup did not survive unscathed. Fixed:
  - **Blocker**: "late-phase" gains cited but stored nowhere + `swiglu_M2048_h1024` (the largest
    predicted gain) had an unmeasurable baseline (unfused < bare GEMM) yet was spun as
    supporting. Fix: derived late-phase fields + `baseline_unmeasurable` flag added to
    `rtx4060_fusion.json:annotations_post_hoc`; row reclassified as no-data.
  - **Major**: "compile never fused 12/12" was false — default compile DID vendor-fuse the
    residual epilogue (4/4, profiler-verified); only "no fused Triton *template*" is true. The
    "no out-of-the-box path" qualifier restricted to F4/grouped-MoE; for F1, addmm/compile-nocg
    are out-of-the-box wins on CUDA (the C500's addmm loss = genuine vendor difference).
  - **Major**: aggregate transparency — all-verified n=10 geomean 0.983 and n=7 sensitivity
    (+1.4% vs est +6.0%) now reported alongside the clean-5 headline; ±0.05 threshold disclosed
    as analysis-time; delivered-fraction geomean 0.78 (range 0.37–1.74) reported.
  - **Major**: template-tax range corrected to 0.96–1.59× (1.59 on a CLEAN config —
    conflation of template and hand-kernel ranges before); hand kernel quoted clean-only
    (0.98–0.99×, +5.6…+7.4%); drift-tainted 0.94× endpoints dropped from headlines.
  - **Major**: numerics claims now backed by data IN the deliverable
    (`annotations_post_hoc.residual_numerics`, fresh seed-0 tensors, labeled post-hoc); the
    GLM-ordering rows honestly show BOTH paths carry comparable cancellation error at k=16384
    (fused-more-accurate holds only on task-ordering dims).
  - Throttled rows reclassified: fused kernel held vendor parity vs the contemporaneous GEMM
    (1.006×/1.047×) — exclusion is an ordering confound, NOT §5-criterion-3 fused-path slowness.
  - Minors: compiled-vs-eager 10/12 (not 12/12), throttle overrun split fused 26–39% / unfused
    3–11%, est ranges labeled per profile, +0.6…+7.4% verified range, T3 within-2× 95.8% added.

## Close-out (2026-07-21)

- Post-review re-audit (independent agent) re-derived every figure in the writeup from the
  JSONs: all headline numbers confirmed; 7 residual slips fixed (rounding 0.833→0.832; the n=7
  sensitivity mislabeled — the honest bracket is now in the writeup and BOTH n=7 aggregates are
  stored in `annotations_post_hoc.aggregates`; the unmeasurable row's self-contradictory
  internal numbers now shown; 0.98–1.01× range; ≈1.07–1.35 GLM numerics low end; all-12 0.992
  added to the JSON; 36-line kernel).
- Deliverables final: `notes/rtx4060_worklog.md` (this file), `rtx4060_measured.json` (T2+T3),
  `rtx4060_fusion.json` (12 configs, task-§4 schema + post-hoc annotations),
  `rtx4060_sim_real_results.md` (verdict A, adversarially reviewed twice). Scripts:
  `rtx4060_common.py`, `rtx4060_peak.py`, `rtx4060_gemm_validate.py`, `rtx4060_fusion_measure.py`.

## Plan (original, for reference)

1. T1 env discovery — done (above).
2. Parallel repo digest (workflow): estimator API, fusion estimator, metax measurement methodology,
   C500 prior results, snowcat_demo import surface.
3. Author measurement scripts: `rtx4060_peak.py` (T2), `rtx4060_gemm_validate.py` (T3),
   `rtx4060_fusion_measure.py` (T4) — all write JSON.
4. Run T2 → T3 → T4 serially (GPU exclusivity; no parallel GPU jobs).
5. T5: estimator-vs-measured comparison, `rtx4060_sim_real_results.md`, adversarial review pass.
