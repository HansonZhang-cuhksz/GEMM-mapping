# RTX 4060 sim–real fusion gap — worklog

# ROUND 3 — T6 (RMSNorm + MoE-structure fusions A–E), LOCKED CLOCKS (2026-07-22)

Task file grew +48 KB with §T6 (tests A–E) — read in full. Host confirms clocks still locked
1500/5501. Host decisions (asked, not assumed): (1) A-epilogue prefill capped at tokens ≤ 32768
(131072 omitted, documented); (2) A-epilogue input INCLUDES a pre-materialized residual
(layer-faithful variant: RMSNorm(mla_o_out + res); the reduction remains the crux, residual-add
is tile-local per F1); (3) run all five tests sequentially in one go (~4–8 h GPU).

Plan: author `rtx4060_rmsnorm_fuse.py` (A), `rtx4060_router_prologue.py` (B), and extend
`rtx4060_fusion_measure.py` + `fusion_time_estimator.py` (C: router top-k attempt-and-drop;
D: FFN L1/L2/L3-F6 levels + `estimate_ffn_fused_m` tpe-parametrized estimator (HARD prereq,
acceptance-checked); E: expert-merge+residual2) via parallel author agents + adversarial
verifiers (round-1 pattern), then run full sweeps serially A→B→C→D→E at the lock, then append
five sub-sections to `rtx4060_sim_real_results.md` + audit. New-test rows for C/D/E MERGE into
the existing locked `rtx4060_fusion.json` (extend, not overwrite; T4 rows + annotations kept).

## R3 prep (2026-07-22)

- Prep workflow hit the session usage limit mid-flight: B author finished (smoke-passed);
  A author + CDE author + B verifier died. Finished the rest inline:
  - **CDE extensions** (agent-partial, completed by me): all C/D/E functions + flags + merge-append
    landed and compile; `estimate_ffn_fused_m` in `fusion_time_estimator.py` passes ALL THREE
    D.5 acceptance checks (0.2591≈0.259 at m=64; mt-scaling 32×; m0 capped 16 by SMEM) and
    reproduces the D.5 predicted table. Fixes applied by me: (1) merge fusion judge missed the
    `triton_per_*` (persistent-reduction) kernel-name prefix — observed
    `triton_per_fused_add_mul_sum_unsqueeze_0` as THE single fused kernel → false negative,
    fixed; (2) added `measured_gain_verified_stock` (compiled/nocg only, no hand kernel/baddbmm)
    as E's out-of-the-box headline metric; (3) enriched the F6 infeasible evidence (below).
  - **F6 (D.4) attempt outcome — documented infeasible in Triton**: 4-chunk kernel (CH=512)
    OutOfResources at every candidate (Required 176–322 KiB vs 101376 B — tl.dot stages every
    operand in SMEM; the paper 84 KiB budget is unreachable without explicit buffer reuse);
    my 8-chunk variant (CH=256, BK=32, s=1) COMPILES within SMEM but returns corrupt results
    (rel 57–252 config-dependent) while the identical phase-1 chunk in isolation is correct
    (rel 0.011) — Triton codegen/SMEM-liveness failure with 8 persistent dot operands.
    → estimator-only F6 per D.4 drop rules i/ii, evidence recorded in the JSON rows.
    (A CUTLASS-class kernel with explicit SMEM management could still hit the budget.)
  - **A script** (`rtx4060_rmsnorm_fuse.py`) authored by me: epilogue (residual-included per
    host decision) with wide-tile attempt protocol + P2/P1 prologue hand kernels + spec-A.5
    est helpers with the structure-blind guard. Smoke-passed end-to-end (epilogue stock
    fusions 0/1 as expected; wide-tile runs at smoke dims where it fits SMEM — will record
    OutOfResources at full dims).
  - B self-reviewed (est formulas verbatim, no-split-K, untimed Wp fold, 131072→65536 OOM
    fallback); its verifier died on the session limit — compensated by the author's smoke
    evidence + my targeted review.

## R3-T6 — full runs complete (2026-07-22, locked clocks verified 1500/5501 before launch)

Chain A→B→C→D→E ran 14:57–15:46 (49 min — warm inductor caches made compiles nearly free).
All exit 0; only `merge_prefill_T49152` errored (OOM at end-of-sweep; then an int32
pointer-overflow in the hand merge kernel at flat index 2.42e9 > 2^31 — kernel fixed to int64
row offsets, row re-measured in a fresh process). Row counts: A 17/17, B 16/16, fusion.json
+9 topk +9 ffn +2 grouped +9 merge (old T4 rows + annotations preserved by merge-append).

Headlines (details in the results file):
- **A epilogue**: 0/7 stock fusions; wide-tile OutOfResources 7/7 (384 KiB fp32 stage vs
  99 KiB); measured gains ≈1.00 — structurally infeasible, exactly as predicted; the
  structure-blind est cell (1.033) stands as the flagged mismatch datapoint.
- **A prologue**: hand P2/P1 numerics-OK everywhere; wins ONLY on the skinny router host
  (+17…+27% vs honest-P2 est +27…+30%); loses on up_gate (0.74–1.01 vs est +0.4–2.6%) — the
  hand GEMM cannot match cuBLAS on wide shapes, eating the small prologue saving.
- **B router prologue**: the study's largest verified gains. b2 clean geomean **1.844**
  (M≥2048, up to 2.17×) vs est 2.230; b1 **1.478** (M≥2048, excl the 131072 row whose 9.79×
  reflects a memory-pressure-degraded baseline: eager 294 ms vs GEMM-only 23.7 ms) vs est
  3.04. Inductor forced-template folded the input add 0/16 — hand kernel is the only fused
  path. Spec caveats apply (router is a minor cost center; shared h/x shrink attribution).
- **C router top-k**: formal **DROP** per C.6 — `router_topk_drop_evaluation`: no stock path
  fused 9/9 (separate topk kernel survives, recorded), custom BN=256 kernel gain ≤1.0 on all
  6 drift-clean configs (0.58–1.00). The est traffic-only bound (~1.03) did not materialize.
- **D FFN levels**: F6/L3 estimator-only (Triton-infeasible, evidence in rows; est cliff
  r_L3 1.005 at mt=1 → 0.15 at mt≥16 stands as prediction). Measured **L2 at real GLM
  per-expert dims is neutral-to-negative** (r_up_swiglu 0.74–1.10, only tpe=64 at +10%;
  r_L2 0.80–1.07) — refines T4: SwiGLU fusion pays on dense mid-size shapes, not in the
  skinny-tpe per-expert regime. Grouped cross-check timings fine (grouped ≤ 8× single,
  0.80–0.97×); grouped rows' fp32-reference numerics field is buggy (rel ~800–1000,
  reference-construction bug — flagged, timings unaffected).
- **E merge+residual2**: the cleanest confirmation — verified gains **1.20–1.28 at every
  token count vs est exactly 12/10 = 1.20** (profile- and token-independent); stock
  `torch.compile` fuses it 8/8 (single `triton_per_fused_*` kernel; judge fixed for that
  prefix); stock-only gains 1.13–1.28. Flatness 512→32768 = 1.2497/1.2423/1.2318;
  49152 point re-measured after the int64 fix (token-independence block updated after).

## R3 close-out (2026-07-22)

- `merge_prefill_T49152` recovered after two failures (end-of-sweep OOM; then an **int32
  pointer-arithmetic overflow in the hand merge kernel** — flat index 49152·8·6144 = 2.42e9 >
  2³¹, fixed to int64 row offsets): verified gain 1.2298 (stock path, fused). Its eager paths
  paged into host memory (12.4 s, WSL2 oversubscription) — gain computed against the clean
  device-resident baseline. **Token-independence CONFIRMED**: 1.2497/1.2423/1.2318/1.2298 at
  512/8192/32768/49152, spread 1.6% ≤ 5% → the infeasible 131072 point is defensibly covered
  (`merge_token_independence` block updated in the JSON).
- Appended the five T6 sub-sections + T6 verdict to `rtx4060_sim_real_results.md`. Independent
  number audit (agent, python re-derivation over all three JSONs + a live estimator re-run for
  the honest-P2 figures): all headline geomeans/tables confirmed; 8 corrections applied —
  notably: the A wide-tile failure is a Triton `CompilationError` (tl.arange needs a power-of-2
  width; N=6144 inexpressible; 512 KiB padded stage rules it out a fortiori), NOT the
  OutOfResources I had assumed; P1 numerics 9/10 (tpe=1024 misses tol); A up_gate clean geomean
  0.87 (n=5); C clean-subset range 0.75–1.00 (0.58 was the tainted M512 row); C recorded est
  cells +1.3…+1.9% (3.5% was the spec ceiling); D tpe=64 isolated est +2.0%; E baddbmm never
  the fastest fused path; T6-verdict stock-compile merge range +13…+28%.
- **T6 deliverables complete**: `rtx4060_rmsnorm_fuse.py` + `rtx4060_rmsnorm.json` (A),
  `rtx4060_router_prologue.py` + `rtx4060_router_prologue.json` (B), extended
  `rtx4060_fusion_measure.py` + `fusion_time_estimator.py::estimate_ffn_fused_m` with C/D/E
  rows merged into `rtx4060_fusion.json` (+ `router_topk_drop_evaluation`,
  `merge_token_independence` blocks), five sub-sections + T6 verdict in
  `rtx4060_sim_real_results.md`, this worklog.

# ROUND 5 — residual₂ → dense DOWN-GEMM epilogue (2026-07-23)

Task: `RTX4060_RESIDUAL_DOWN_TASK.md` — the one untested residual site in the dense case:
`out = addmm(residual2, x_act, W_down)` at N=HIDDEN=6144, K-sweep {2048, 6144, 12288, 16384,
24576} (headline K=24576 = 4×H dense FFN width), M-sweep {512…16384, 32768, 49152}
(131072 dropped with arithmetic; covered by per-K M-independence assertion at M∈{512,8192,
32768}). Both comparisons per the Round-4 lesson: vendor-stock (PRIMARY — addmm is expected to
fuse stock) and custom-vs-custom (mechanism isolation). Estimator cell VALID (tile-local
epilogue); sanity anchors ≈1.159 @ (2048,2048), ≈1.013 @ (2048,24576) to reproduce.

Host decision (asked): **core-full + light rest** sweep — all 6 paths (incl. the 3
torch.compile variants) on the 15 core configs (5 K × M∈{2048,8192,32768}); the other 25 rows
run the fast paths only (unfused incl. compile-of-add clean variant, addmm + kernel evidence,
custom pair) with compile-path fields null-with-reason (~120 fresh GEMM autotunes avoided;
answers §9 fully). Locked clocks re-verified before run.

## R5 results + close-out (2026-07-23)

- `rtx4060_residual_down.py` authored (est wiring verbatim from spec §3 — sanity anchors
  reproduced exactly: 1.1588/1.0132; custom kernel family per §4; core-full + light-rest sweep
  per host decision), smoke-passed, full run at the verified lock: **40/40 rows, 0 errors**
  (~2 h). `rtx4060_residual_down.json`.
- **Q1 answered: `addmm` fuses residual₂→down stock, 40/40** (profiler-verified single GEMM,
  numerics OK) → `needs_custom_kernel = False` everywhere — the canonical case confirmed.
- **Three-metric story** (drift-clean n=31): (1) mechanism (custom-vs-custom same-tile) delivers
  the estimator's K-law — per-K 1.114/1.047/1.020/1.019/1.022 vs est 1.159/1.053/1.027/1.020/
  1.013, delivered 0.72–1.64, overall 0.82; (2) stock addmm fuses but delivers only ≈0.25
  overall (+1.2% vs est +4.9%) and is NEUTRAL at the K=24576 headline (0.995; cuBLASLt's addmm
  kernel choice is 5–12% worse than its mm choice on 4 rows — vendor-heuristic tax); (3)
  `measured_gain_verified` inflated at M=32768 K≥16384 by a GEMM-quality INVERSION — the forced
  Triton template beats cuBLAS mm by 16–19% there (forced/vendor 0.81–0.84) — flagged, and the
  clean addmm framing added to the JSON as `aggregate.addmm_primary` (+ per-row `addmm_gain`).
- M-independence: flat (spread ≤0.2%) at K∈{12288,16384}; not flat at K=2048 (small-M
  occupancy), K=6144 (addmm-slow 8192 row), K=24576 (`r5_M49152_K24576` ran under WSL2 memory
  paging — vendor GEMM 13.9 s vs expected ~0.8 s — excluded as unusable; M=32768 inversion row).
  131072 coverage claimed only via the mid-K flatness + K-driven structure for M≤16384.
- Bracket check vs R2-F1: K=6144 addmm +1.3% / K=16384 +1.0% vs F1's +1.4…+2.1% / +0.3…+0.7% ✓.
- Addendum + main-verdict forward-pointer written into `rtx4060_sim_real_results.md`.

# ROUND 4 — N4 CUSTOM-vs-CUSTOM ADDENDUM (2026-07-22)

Host request: in the N4 (SwiGLU→up_gate, F4) step — both dense and MoE — the fused custom
Triton kernel was previously judged against vendor baselines (cuBLAS GEMM + eager/compiled
epilogue), so the measured "fusion gain" conflated (fusion benefit) − (Triton-vs-cuBLAS
GEMM-quality gap). Re-compare against a **custom UNFUSED implementation of the same kernel
family only**. Host decisions (asked): config scope = T4 dims (dense 6 + MoE grouped E∈{8,32})
AND T6.D GLM per-expert dims (tpe sweep + grouped E=8); baseline = fully custom 2-kernel
(Triton GEMM writing full [M,2·inter] gu + separate custom Triton SwiGLU elementwise — no
vendor code anywhere in the comparison).

Plan: new `rtx4060_n4_custom.py` → `rtx4060_n4_custom.json`; three batched-capable Triton
kernels sharing one tiling family (plain full-width GEMM / SwiGLU elementwise / dual-
accumulator fused); per config report (a) best-vs-best custom gain, (b) same-tile gain
(identical (BM,BN,BK,w,s) on GEMM-full and fused — pure mechanism isolation), plus vendor
references (cuBLAS gemm, eager unfused) as context only; est gain from est_swiglu_ms (the
estimator's model assumes same-quality GEMM on both sides — custom-vs-custom is exactly its
regime, making this the cleanest est test of the study); fp32-ref numerics both paths; drift
probe = custom GEMM re-measured at config end; locked clocks verified before run. Then a
results-md addendum sub-section + audit.

## R4 results + close-out (2026-07-22)

- `rtx4060_n4_custom.py` authored (3 batched-capable Triton kernels sharing one tiling family:
  full-width GEMM / SwiGLU elementwise / dual-accumulator fused), smoke-passed, full run at the
  verified lock: 18/18 configs, numerics 18/18 both paths. `rtx4060_n4_custom.json`.
- **Custom-vs-custom results (the host's requested comparison):** all-18 geomeans — measured
  **1.045 (best-vs-best) / 1.061 (same-tile) vs est 1.051**; positive in 15/18 (best) and
  14/18 (same-tile); per-group (all rows): t4_dense 1.069/1.090 vs est 1.091; t4_moe (first
  custom fused grouped path ever) 1.026/1.054 vs est 1.079; glm_dense 1.028/1.033 vs est
  1.021; glm_grouped 1.060/1.095 vs est 1.023. Custom GEMM ran 0.86–1.24× vendor (gm 1.02) —
  the quality gap that contaminated the earlier vendor-baseline framing.
- **Study-level consequence** (added to the results verdict + addendum §): the T4 headline
  "estimator over-predicts ~2×" decomposes into (vendor-vs-custom kernel-quality gap) + a
  small residual model error — delivered fraction ≈ 0.88–1.20 in the estimator's own
  (same-kernel-family) regime. N4 fusion is genuinely profitable within a fixed kernel family,
  including the GLM per-expert and grouped regimes that looked negative against vendor
  baselines. Deployment rule sharpened: fuse iff you already ship a custom kernel or match
  vendor GEMM quality — which is exactly the C500/MACA-CUTLASS situation, where the estimator
  was right all along.
- Drift caveat: heavy per-config tuning load → only 5/18 strictly clean rows; relaxed ≤1.10
  subset (n=10) reads 1.073/1.063 vs est 1.064 (same agreement). Appended "N4 addendum"
  section to `rtx4060_sim_real_results.md` + forward-pointer from the main verdict.

# ROUND 2 — LOCKED-CLOCK RE-RUN (2026-07-22)

The host locked the clocks to **1500 MHz core / 5501 MHz VRAM** (the exact calibration point of
the `rtx4060-measured` profile) and asked for a full T1–T5 re-run. Per their choice: unlocked
results archived as `*_unlocked.*`; fresh canonical deliverables from locked-clock runs; the
writeup's verdict now rests on locked-clock data with a locked-vs-unlocked comparison.

## R2-T1 — environment re-check (locked clocks)

- `nvidia-smi`: gr 1500 MHz / mem 5501 MHz at idle AND rock-solid under 3 s of sustained GEMM
  load (every 0.2 s sample identical: "1500, 5501"); persistence mode Enabled; 57 °C idle.
- Quick probe: 4096³ bf16 matmul = 7.627 ms = **18.02 TF/s** at the lock — consistent with the
  profile's calibration (peak_tensor_flops 18.43 = 96 TC × 128 × 1.5 GHz).
- Device props unchanged from R1-T1 (24 SM, 8 GiB, cc 8.9, 99 KiB opt-in SMEM, 32 MiB L2);
  torch 2.11.0+cu130 / triton 3.6.0 / py3.13.
- Methodology note: `warm_clocks()` stays enabled in `med_time` — a semantic no-op at locked
  clocks, kept as a guard in case the lock drops mid-run (WSL2); drift probes and ClockSamplers
  stay too, now serving as lock-held verification.

## R2-T2 — peak specs at locked clocks (2026-07-22)

- **18.25 TF/s (0.990× profile) and 167.9 GB/s (0.988×)** — the `rtx4060-measured` profile
  reproduces to within ~1.2% at its own calibration point. Implied tensor clock 1485 MHz vs the
  1500 lock. Adjusted profile ≈ stock (clock 1.485e9, bw 1.679e11) — carried through anyway for
  consistency. `rtx4060_peak.json`.
- Contrast unlocked R1: 18.79 TF/s (1.019×) / 214.8 GB/s (1.264×) — the unlocked memory clock
  was the big deviation.

## R2-T3 — estimator validation at locked clocks (2026-07-22)

- 24 shapes: **geomean est/meas = 0.943 stock / 0.952 adjusted; 96%/100% within 1.5×;
  100% within 2×; ratios span 0.663–0.991** → IN the 0.72–0.96 band, and the estimator is now
  *uniformly* mildly optimistic (no ratio > 1 — R1's pessimistic ratios up to 1.15 were
  boost-clock artifacts). FFN subset: 0.948, 100% within 1.5×.
- Every R1 flaky shape resolved: 2048×1024×1024 now 0.309 ms / 13.9 TF/s in-sweep (R1: 0.73 ms /
  5.8 TF/s); achieved TFLOP/s across the sweep is a tight 16.2–18.3 for all non-tiny shapes.
  DVFS was conclusively the R1 noise source. `rtx4060_gemm_validate.json`,
  merged → `rtx4060_measured.json`.

## R2-T4 — launched (2026-07-22)

- Same protocol as R1 take 2 (12 configs, all paths, --moe, --t2-json), locked clocks.
  Log: scratchpad/t4_locked_run.log.

## R2-T4 — results (2026-07-22)

- 12 configs completed at the lock; `rtx4060_fusion.json` (+ `annotations_post_hoc` with
  residual numerics, late-phase fields, aggregates). Lock held in 11/12 configs (all ClockSampler
  medians 1500); `residual_glm_M8192` sagged to 1290 MHz median — **power-limit throttling below
  the lock** (35 W part): `-lgc` caps but cannot floor. Its sibling glm_M2048 drifted 1.13.
- **All 6 swiglu configs drift-clean** (0.98–1.02): verified gains 1.062/1.024/0.993/1.108/
  1.088/0.974 vs est 1.158/1.079/1.040/1.159/1.079/1.041. Hand kernel 1.03–1.08× vendor.
  The R1 throttle-excluded `swiglu_M8192_h2048` now reads **1.088 vs est 1.079** — R1's
  exclusion vindicated.
- Residual: clean M2048 config **matches est exactly** (forced-template gain 1.052 vs est 1.053
  at 1.0005× vendor). M8192: addmm +1.4%, forced template loses parity (1.13×). GLM rows
  power-dip-tainted (drift 1.13/1.23) — glm_M2048's addmm 0.888 contradicts its own R1 value
  (1.006), flagged unreliable.
- **Aggregates**: verified-gain geomean 1.0285 (n=10) / 1.0419 (clean n=7) vs est 1.0692/1.0860
  → **delivered fraction ≈ 0.41/0.49** — the estimator over-predicts magnitude ~2× at these
  dims, now cleanly measurable because clocks are locked (stock ≈ adjusted profile: 18.25 TF/s,
  167.9 GB/s). Direction/ranking correct; C500-convention all-12 geomean now positive (1.061).
- Hypotheses for the over-prediction (in writeup §4): (i) eliminated activation traffic partly
  L2-resident at small h (est charges DRAM bw; over-prediction largest at h=1024), (ii) at
  h=4096 the hand kernel's 1.06–1.08× overhead eats the small (+4%) predicted gain. The exact
  est-match at the vendor-parity residual config supports (ii).

## R2-T5 — verdict (2026-07-22)

**A upheld on locked-clock data, with the quantitative caveat now front-and-center**: fusion
gains are real on verified-fused paths (+2.9% geomean n=10, up to +10.8%; fused kernels
1.00–1.08× vendor — no C500-style tax), but realized magnitude ≈ 0.4–0.5× the estimator's
prediction. Rewrote `rtx4060_sim_real_results.md` (locked-clock canonical, lock stated
prominently per host request, §7 locked-vs-unlocked comparison). Unlocked round archived as
`*_unlocked.*`.

## R2 close-out (2026-07-22)

- Independent number re-audit of the locked-clock writeup vs all four JSONs (+ R1 archives for
  the §7 comparison): headline numbers all confirmed; 5 slips fixed (MoE +3.0%, drift 1.22,
  compile beats eager in 1/12 with 1 tie, over-prediction-largest claim restricted to the
  M2048-h1024 single max, gu range 8–67 MB). Stale `clocks_locked=false` metadata (hardcoded R1
  note in the scripts) corrected post-run in `rtx4060_fusion.json` and `rtx4060_measured.json`,
  each with an explicit [corrected post-run] marker.
- Deliverables (locked-clock canonical): `rtx4060_measured.json`, `rtx4060_fusion.json`,
  `rtx4060_sim_real_results.md`, this worklog. R1 unlocked archives: `rtx4060_measured_unlocked.json`,
  `rtx4060_fusion_unlocked.json`, `rtx4060_peak_unlocked.json`, `rtx4060_gemm_validate_unlocked.json`,
  `rtx4060_sim_real_results_unlocked.md`.

(Round-1 unlocked-clock log follows below.)

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
