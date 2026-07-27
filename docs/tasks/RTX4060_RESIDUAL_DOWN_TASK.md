# TASK — RTX 4060 (Round 5): residual₂ fused into the **dense down-projection GEMM** epilogue

> Standalone follow-up to `RTX4060_SIM_REAL_TASK.md`. **Reuse its shared preamble verbatim**
> (§T6.0.0–T6.0.5: `rtx4060_common` timing/drift/verify/numerics infra, locked-clock protocol,
> `build_adjusted_profile`, the 6-path taxonomy). This spec only defines the ONE new fusion, its
> dims, and its estimator wiring. Locked clocks **1500 MHz core / 5501 MHz VRAM** as in Rounds 2–4.

## 0. The one question to answer

Every residual fusion measured so far was at a *different* site:
- **F1 / `residual`** = residual**₁** → `mla_o` (attention out-proj), shapes `(N=16384,K=6144)` and
  `(N=6144,K=16384)` — proved cuBLASLt `addmm` β-accumulate fuses a residual into a GEMM epilogue,
  stock, delivering +5.2% / +1.4% … ~1.0 depending on K.
- **E / `merge_r2f`** = residual**₂** → the MoE **expert-combine** reduction (`baddbmm`, K=top_k=8),
  memory-bound, +24%.

**Neither is the dense residual₂ site.** In a **dense** transformer there is no expert-combine, so
residual₂ must fold into the **down-projection GEMM** epilogue:
`out = residual2 + x_act @ W_down` = `addmm(residual2, x_act, W_down, β=1)`. This is the *same*
cuBLASLt β-accumulate mechanism F1 validated, but at the **down shape** `(N=HIDDEN, K=INTERMEDIATE)`,
which was never run. **Question: does residual₂→down deliver the estimator's predicted gain, and is it
stock-fusable (`addmm`) with no custom kernel — i.e. the cleanest verdict-A datapoint of the study?**

**Deliverable:** a per-(M,K) est-vs-measured table with delivered fractions, drift-clean, both a
vendor-stock baseline and a custom-vs-custom baseline (the Round-4 lesson), and a one-line verdict.

## 1. The fusion + exact dims (dense FFN convention)

Dense FFN (NOT MoE — every token flows through ONE down GEMM, so **`M = tokens` directly**, no `/32`):

```
x_act = SwiGLU(up_gate(x))          # [M, K]  activations, K = INTERMEDIATE (dense width)
y     = x_act @ W_down              # down GEMM: [M,K] @ [K, N] -> [M, N], N = HIDDEN = 6144
out   = residual2 + y               # residual2 : [M, N]  ->  fold into the down epilogue
```

| tensor | shape | dtype |
|---|---|---|
| `x_act` (down input) | `[M, K]` | bf16 |
| `W_down` | `[K, N=6144]` | bf16 |
| `residual2` | `[M, N=6144]` | bf16 |
| `out` | `[M, N=6144]` | bf16 |

`N = HIDDEN = 6144` fixed. **The residual add is TILE-LOCAL** (`out[m,n] = residual2[m,n] + y[m,n]`,
same index as the GEMM output tile — NO cross-tile reduction; contrast F4 SwiGLU's disjoint column
slices and A/RMSNorm's over-N reduction). So this fusion is template-expressible AND `addmm`-expressible
→ the estimator cell is **VALID (not structure-blind)**, and the stock path is expected to fuse.

### 1.1 K-sweep (the dense width axis — this is what makes the answer K-dependent)

The residual round-trip is a fixed `~2·M·N·BPE` saving; the down GEMM cost grows with K. So the *fraction*
saved (the gain) falls as K rises. Sweep K to span dense widths and **bracket the two F1 points**:

| K | meaning | est gain (rtx4060-measured, precomputed) |
|---|---|---|
| 2048 | narrow / GLM per-expert width | **~1.159** (memory-bound down GEMM) |
| 6144 | = HIDDEN (F1 `residual_M` bracket) | ~1.05 |
| 12288 | 2×HIDDEN | ~1.02–1.03 |
| **24576** | **4×HIDDEN — standard dense FFN width (HEADLINE)** | **~1.013** (compute-bound) |
| 16384 | = KV (F1 `residual_glm` bracket) | ~1.02 |

Headline dense case = **K=24576**. Keep the full sweep so the K-dependence (and the bracket vs F1)
is visible. (These est numbers were computed with the §3 snippet on `rtx4060-measured`; reproduce them.)

### 1.2 M-sweep + regimes (reuse T6.0.3 shape, but `M = tokens` for the dense down GEMM)

- **DECODE:** `M ∈ {512, 1024, 2048, 4096, 8192, 16384}` (`regime:"decode"`)
- **PREFILL:** `M ∈ {32768, 49152}` (`regime:"prefill"`), keep `8192` as the decode/prefill boundary.
- **`M = 131072` is INFEASIBLE at 8 GB for the wide-K rows — DROP with arithmetic:** at K=24576,
  `x_act[131072,24576]` bf16 alone `= 131072·24576·2 = 6.4 GB`; plus `W_down`, `out`, `residual2`,
  fp32 ref → OOM. Practical ceiling ≈ `M=49152` at K=24576 (`x_act ≈ 2.4 GB`, total ≈ 4–5 GB — fits;
  free the fp32 ref + unfused `y` intermediate before allocating). For narrow K (2048/6144) the top
  prefill point may go higher, but keep the sweep uniform at `≤49152` for cross-K comparability and
  record the drop. Gain is K-driven, not M-driven (see §5), so the missing 131072 point is covered by
  the M-independence of the ratio — **assert it** (measure gain at M∈{512, 8192, 32768}, confirm flat
  per K, record `gain_token_independent_perK`).

## 2. Two comparisons (BOTH required — the Round-4 lesson)

**(a) vs VENDOR (the deployment answer — PRIMARY here, because this fusion is stock):**
- `unfused` = cuBLAS `torch.mm(x_act, W_down)` **then** a separate residual add (eager `y + residual2`,
  and the `torch.compile`-of-add clean variant); `best_unfused_ms` = min over those.
- `addmm` = `torch.addmm(residual2, x_act, W_down)` (cuBLASLt β=1 accumulate) — **the canonical fused
  path.** Expect it to FUSE and be the (or a) verified-fused winner. This is the whole point: unlike
  SwiGLU (F4), residual₂→down needs **no custom kernel** — vendor `addmm` captures it.
- Also run `compiled` / `nocg` / `forced` / `triton` per the 6-path taxonomy (below).

**(b) CUSTOM-vs-CUSTOM (the estimator's own regime — isolates the mechanism from the GEMM-quality gap):**
- `custom_unfused` = plain full-width custom Triton GEMM + a separate custom residual-add kernel (2
  kernels, same Triton tiling family, NO vendor code).
- `custom_fused` = the same Triton GEMM with the residual read folded into its store epilogue (§4).
- Report `custom_gain_best` (each side independently tile-tuned) and `custom_gain_same_tile` (unfused
  GEMM forced onto the fused kernel's exact `(BM,BN,BK,warps,stages)` — pure mechanism isolation),
  exactly as `rtx4060_n4_custom.py`. Context columns: `vendor_gemm_only_ms`, `vendor_eager_unfused_ms`,
  `custom_gemm_over_vendor_gemm` (quality gap). **Reuse the `rtx4060_n4_custom.py` harness** — this is
  the same shape of experiment with a residual epilogue instead of a SwiGLU epilogue.

## 3. Estimator wiring (VALID cell — reproduce these exactly)

```python
import fusion_time_estimator as fte
from fusion_time_estimator import (Epilogue, _residual_aux,
                                    estimate_fused_gemm, estimate_gemm_grouped, estimate_vector_kernel)
HIDDEN, BPE = fte.HIDDEN, fte.BPE   # 6144, 2

def est_res2_down_ms(M, K, gpu, N=HIDDEN):
    g   = estimate_gemm_grouped("down", M, N, K, 1, gpu).time_s          # bare down GEMM
    res = estimate_vector_kernel("res2", 3*M*N*BPE, gpu).time_s          # standalone add: 3·M·N traffic
    unf = g + res
    fus = estimate_fused_gemm("down+res2", M, N, K, 1,                    # residual read folded in
              Epilogue(extra_hbm_once=M*N*BPE, aux_smem_per_tile=_residual_aux), gpu).time_s
    return unf*1e3, fus*1e3, unf/fus                                      # est_unfused_ms, est_fused_ms, estimated_gain
```

Report the trio for both the **stock** profile and the **`--t2-json` adjusted** profile
(`build_adjusted_profile`). This is a real GEMM with a tile-local epilogue → the estimator models it
correctly; **do NOT flag structure-blind** (contrast A-epilogue / C). Sanity: at `(M=2048,K=2048)` you
should get `estimated_gain ≈ 1.159`; at `(M=2048,K=24576) ≈ 1.013`.

## 4. Custom fused kernel (the down GEMM with a tile-local residual epilogue)

```
# out-tile [BM × BN]; standard Triton GEMM over K, then fold residual2 into the store
acc = tl.zeros((BM, BN), tl.float32)
for k0 in range(0, K, BK):
    a = tl.load(x_act  + rows[:,None]*K + (k0+kcols)[None,:])     # [BM,BK]
    b = tl.load(W_down + (k0+kcols)[:,None]*N + cols[None,:])     # [BK,BN]
    acc += tl.dot(a, b)
acc += tl.load(residual2 + rows[:,None]*N + cols[None,:])        # [BM,BN] tile-local residual add
tl.store(out + rows[:,None]*N + cols[None,:], acc.to(out.dtype.element_ty))
```
Tune `(BM,BN,BK,warps,stages)` with the `rtx4060_n4_custom.py` mini-autotuner; numerics vs fp32 ref,
`rel_max ≤ max(2·eager_rel, 5e-2)`. `custom_unfused` = the identical GEMM **without** the residual add +
a separate residual-add kernel; `custom_unfused_same_tile` = that GEMM forced to the fused tile.

## 5. 6-path verdict (expected — the canonical stock-fusable case)

| path | expected | note |
|---|---|---|
| `unfused` | baseline | `mm` + separate add (2 kernels) |
| `addmm` | **FUSES ✓ (primary)** | cuBLASLt β-accumulate; the textbook GEMM+residual — verified-fused winner |
| `compiled`/`nocg` | likely FUSES | inductor folds `+residual2` into the mm epilogue (or lowers to `addmm`); track `cudagraph_input_copy_us` |
| `forced` | FUSES (template) | residual is tile-local → the Triton GEMM template CAN express it (unlike SwiGLU) |
| `triton` | FUSES (by construction) | the §4 hand kernel = the custom-vs-custom fused path |

**`needs_custom_kernel = False` expected** (like E-merge, unlike F4-SwiGLU): the vendor `addmm`
captures the fusion. `measured_gain_verified = best_unfused_ms / best VERIFIED-fused` (expect `addmm`
or `forced` to be the verified winner).

## 6. Expected outcome (predicted, not measured)

- **Stock/vendor:** `addmm` fuses and delivers ≈ the estimate at **~vendor-GEMM speed** (no quality gap
  on the vendor path) → delivered fraction ~1.0. Magnitude is **K-driven**: ~**+1.3%** at the dense
  headline K=24576 (compute-bound; residual round-trip is a tiny fraction), rising to ~**+16%** at
  narrow K=2048 (memory-bound). Brackets the F1 points (K=6144→~+5%, K=16384→~+1%).
- **Custom-vs-custom:** mechanism delivers the estimate (same-tile ≈ `estimated_gain`), as in the N4
  addendum — but here the *vendor* path already fuses, so the custom route is confirmatory, not required.
- **Verdict-A reading:** a memory-bound-leaning residual fusion, **stock-fusable via `addmm`**, delivering
  the predicted gain with no custom kernel — the dense counterpart to E-merge. The dense-model takeaway:
  at realistic FFN width (4×H) the residual₂→down saving is **small (~+1%) but free**; it only becomes
  large when the FFN is narrow.

## 7. Caveats to honor (attempt-and-drop)

- **Small decode M (512, 1024)** at wide K → few row-tiles on 24 SM, latency/occupancy-bound; flag rows
  with mutually-inconsistent `gemm_only`/`unfused` and `|drift−1|>0.05`, exclude from the geomean (prefer
  `BM=64` to raise tile count). Keep in JSON.
- **Drift probe:** re-measure the bare down GEMM (`mm`) at each config's END; `drift_clean` iff
  `|drift−1| ≤ 0.05`; relaxed subset `≤ 0.10` (report both, since Round-4 tuning load left few strictly
  clean rows — same convention). Aggregate/geomean over drift-clean only; also report the relaxed n.
- **`M=131072` dropped** (§1.2 arithmetic) — cover via the per-K M-independence assertion, don't leave silent.
- If `addmm` unexpectedly does NOT fuse on this torch build (it should — canonical case), document the
  profiler evidence and fall back to `forced`/`triton` for `measured_gain_verified`; note the surprise.

## 8. Deliverables (write to `GEMM-mapping/`)

1. **`rtx4060_residual_down.py`** — the measurement script (reuse `rtx4060_common` + the F1 `residual`
   path from `rtx4060_fusion_measure.py` + the `rtx4060_n4_custom.py` custom-vs-custom harness).
2. **`rtx4060_residual_down.json`** — `conventions` + `env` (mirror `rtx4060_n4_custom.json`) + `configs`
   list, one row per `(M,K)` with at least:
   ```
   name, regime, dims:{M,N,K},
   best_unfused_ms, addmm_ms, compiled_ms, nocg_ms, forced_ms, triton_ms,
   best_fused_ms, fused_verified, kernel_evidence, needs_custom_kernel,
   measured_gain, measured_gain_verified,
   custom_unfused_ms, custom_fused_ms, custom_gain_best, custom_unfused_same_tile_ms, custom_gain_same_tile,
   vendor_gemm_only_ms, vendor_eager_unfused_ms, custom_gemm_over_vendor_gemm,
   est_unfused_ms, est_fused_ms, estimated_gain, est_unfused_ms_adj, est_fused_ms_adj, estimated_gain_adj,
   numerics, gemm_drift_ratio, drift_clean, clocks, excluded_from_aggregate
   ```
   Plus a top-level `aggregate` with per-K and overall geomeans (drift-clean and relaxed≤0.10), the
   delivered fractions (`(meas-1)/(est-1)`), and `gain_token_independent_perK`.
3. **"Residual₂→down (dense) addendum"** section in `rtx4060_sim_real_results.md` — the est-vs-measured
   table (per K, per M), vendor + custom-vs-custom, drift caveat, and the verdict; **add a forward-pointer
   from the main verdict** so the dense residual₂ answer is discoverable (as the N4 addendum did).
4. **Round 5** step logs in `notes/rtx4060_worklog.md`.

## 9. What to report (the answer)

1. Does `addmm` fuse residual₂→down (stock, verified)? → `needs_custom_kernel` verdict.
2. Per-K measured gain vs estimate + delivered fraction; does it match the estimator (valid cell)?
3. The dense-headline number (K=24576): the realistic dense-FFN residual₂→down benefit.
4. Confirmation that it brackets the two F1 points across K, and that the mechanism (custom-vs-custom)
   agrees — closing the one untested residual site in the dense case.
