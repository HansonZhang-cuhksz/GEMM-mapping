# RTX 4060 round-2 (locked-clock) result check — log

Checked the results the RTX-4060 machine logged (07-22). Steps:

1. **Scoped the files.** New locked-clock results: `rtx4060_{peak,gemm_validate,measured,fusion}.json`
   + `rtx4060_sim_real_results.md` (round-1 archived as `*_unlocked.*`). Timestamps 07-22_02:23.

2. **PROBLEM found — T6 was NOT run.** `rtx4060_fusion.json` contains only the original T4 kinds
   (`swiglu`, `residual`, `moe_E8/E32`); no `router_topk`/`ffn`/`merge` rows; the T6 output files
   `rtx4060_rmsnorm.json` / `rtx4060_router_prologue.json` are absent. The machine **re-ran the
   original T4 study under LOCKED clocks (1500/5501)** instead of running the T6 additions
   (RMSNorm→mla_o, router prologue, top-k, MoE FFN/F6, ResAdd₂→merge) from `RTX4060_SIM_REAL_TASK.md §T6`.

3. **Verified the locked-clock T4 re-run is SOUND.** T3 reproduces the profile at its calibration
   point (geomean est/meas 0.943; peak 18.25 TF/s = 0.99× profile; HBM 167.9 GB/s = 0.99× profile).
   Aggregates in the JSON match the results doc exactly: drift-clean n=7 measured 1.0419 / est 1.086
   → delivered 0.49; all-verified n=10 → 0.41; swiglu-6 → 0.44. 7/12 drift-clean at 1500 MHz (all 6
   SwiGLU clean). No fabrication; every number traces to the raw JSON.

4. **The revision (important).** At the LOCKED calibration point the estimator over-predicts the fusion
   MAGNITUDE ~2× (delivered ≈0.41–0.49), sharper than round-1's ~0.78 — which was a DVFS artifact:
   the unlocked memory clock (215 vs 168 GB/s) shrank the *predicted* gain toward the measured one,
   masking the over-prediction. Verdict A (fusion real, right direction/ranking, fused kernel at
   1.03–1.08× vendor GEMM, no fusion tax) is UNCHANGED; only the magnitude sharpens (~2× optimistic).

5. **Integrated** into `sim_real_synthesis.md` (RTX-4060 row, verdict, bottom-line, caveats) + flagged
   T6 as still-pending there.

## Open: T6 still needs a run
The spec `RTX4060_SIM_REAL_TASK.md §T6` is ready. The RMSNorm-epilogue feasibility (designed to be
infeasible on 99 KiB SMEM → S3-placement question), the F6 crossover, and the merge (r2f) fusion
remain untested on hardware.

---

# T6 results check (round 3 — T6 run under locked clocks)

The RTX-4060 THIS time ran the full T6 suite (locked 1500/5501). Files present:
rtx4060_rmsnorm.json (A), rtx4060_router_prologue.json (B), rtx4060_fusion.json grew to 41 configs
(new kinds: router_topk, ffn_levels, ffn_grouped, merge_r2f), rtx4060_sim_real_results.md +§T6.A–E.

Inline spot-check (all match the doc; data is real):
 - **A RMSNorm→mla_o**: epilogue STRUCTURALLY INFEASIBLE confirmed (est flagged structure-blind, excluded);
   prologue P2 works, +26.8%/+17.1% on the skinny router host ≈ est. → S3 placement NOT buildable; S5 (prologue) is.
 - **B router prologue**: works; delivered ≈0.83. (verdict quotes up to +117% — a standalone-microbenchmark number.)
 - **C top-k epilogue**: measured 0.78–0.85 (<1, loses) + fused_verified=False → DROP condition met (as predicted).
 - **D MoE FFN**: L2 (SwiGLU) MIXED — 0.804@tpe16 (LOSS), 1.073@tpe64, ~0.99@large. F6: f6_fused_verified=False
   everywhere → estimator-only; est_r_L3 crossover 1.005→0.259→0.15 (tpe-parametrized patch WORKED). Needs CUTLASS.
 - **E merge r2f**: measured 1.23–1.28 vs est 1.20 → delivered ~100–124% (OVER-delivers), stock torch.compile
   fuses it, NO custom kernel. Cleanest verdict-A datapoint. (T49152 fused_verified=False; T131072 OOM-dropped.)

Launched adversarial per-test verification workflow (wko09f7vl). Minor items to watch: merge_prefill_T49152
unverified; D L2 small-tpe loss (real GLM decode dims); router-prologue large % = standalone artifact.

## Verification verdict (workflow wko09f7vl) — SOUND data, writeup problems

All 5 tests: numbers_match_json=TRUE (no fabrication; every figure reproduces). Predictions held
(A/B/D held-with-nuance, C/E held). Problems found (writeup/JSON handling, NOT the core verdict):
 - HIGH (B): JSON aggregate.b1 (1.860) INFLATED by an un-excluded outlier router_b1_M131072 (gain 9.79x,
   a degraded eager baseline: unfused 293.7ms vs 23.7ms GEMM, ~6.4GB buffers on 8GB). excluded_from_aggregate
   =False despite writeup calling it excluded (exclusion test only fires when unfused FASTER, not 12x slower).
   JSON aggregate disagrees with the doc's honest 1.478. Downstream JSON consumers get a wrong b1.
 - MEDIUM (D): the highlighted SwiGLU L2 "+10% win" (tpe64) is a DRIFT-THROTTLE ARTIFACT (only drift-unclean
   row, drift 1.116). ALL 7 drift-clean rows r_L2<=0.990 (ZERO clean wins); grouped cross-check 0.974 loss.
   => SwiGLU fusion NEVER cleanly wins at real GLM per-expert M. My earlier "1.073@tpe64 win" was this
   tainted row — CORRECTED. Conclusion (neutral-to-negative) unchanged/strengthened.
 - MEDIUM (A): router M32768 +17.1% is drift-tainted (drift 1.22) but framed as a clean confirmation.
 - MEDIUM (B): verdict line "+27..+117%" for router prologue lacks the standalone-microbench caveat inline.
 - MEDIUM (E): "8/8 profiler-verified" doesn't disclose 2 prefill points are nocg-verified only (cudagraph
   fused_verified=False; skipped for OOM). Legit but undisclosed.
 - LOW: honest-P2 est reconstructed (not raw JSON field); topk cites 1 of 3 tainted rows; E delivered is
   100-107% (my "124%" framing was slightly high; writeup's "~100%+" is correct).

Verdict UNCHANGED + sharpened -> three-tier realizability hierarchy (memory-bound ~1.0 / GEMM-epilogue ~0.5
& can-lose / cross-tile unreachable). Integrated into sim_real_synthesis.md.
