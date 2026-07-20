"""EXPERIMENT 2 — Is there ANY regime where fused best-throughput beats unfused best-throughput?

Fused's throughput cap comes from weight re-reads: F6 holds activated[m0,INTERMEDIATE] in SMEM,
m0_max = (SMEM - stream_overhead)/(INTERMEDIATE*bpe + 256). Once tokens/expert Me exceeds the
power-of-2 m0_max, mt=Me/m0>1 -> weights re-read mt x -> time ∝ B -> throughput flat/capped.

We test three levers that raise m0_max or change the weight/activation balance:
 (a) INTERMEDIATE width in {512,1024,2048,4096}
 (b) SMEM per block  x{1,2,4} on H100
 (c) DENSE (experts=1, Me=B) vs the MoE baseline

Reuses batch_sweep's validated unfused_layer / fused_layer / set_batch by monkeypatching the
INTERMEDIATE / EXPERTS / TOP_K globals in BOTH modules, then re-running the batch sweep.
All numbers are ANALYTICAL snowcat-roofline ESTIMATES.
"""
import dataclasses
import json
import math

import fusion_time_estimator as fte
import batch_sweep as bs
from gemm_time_estimator import GPUS

H100 = GPUS["h100-sxm"]

# Batch grid: multiples of 32 (so MoE Me=B/32 is integer) + smaller ones for the dense case.
BATCHES = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]


def configure(intermediate, experts, topk, hidden=6144):
    """Point BOTH modules' globals at a workload config, then let set_batch rederive shapes."""
    fte.INTERMEDIATE = intermediate
    fte.EXPERTS = experts
    fte.TOP_K = topk
    fte.HIDDEN = hidden
    bs.INT = intermediate
    bs.EXPERTS = experts
    bs.TOPK = topk
    bs.HIDDEN = hidden


def run_sweep(gpu, batches=BATCHES):
    rows = bs.sweep(gpu, batches)                # uses set_batch/unfused_layer/fused_layer
    ubest = max(rows, key=lambda r: r["unf_tput"])
    ffeas = [r for r in rows if r["fus_tput"]]
    fbest = max(ffeas, key=lambda r: r["fus_tput"]) if ffeas else None
    return rows, ubest, fbest


def m0max_pow2(intermediate, smem):
    raw = (smem - 16 * 1024) // (intermediate * fte.BPE + 256)
    pw = 1
    while pw * 2 <= raw and pw * 2 >= 1:
        pw *= 2
    return raw, pw


def show(tag, rows, ubest, fbest, extra=""):
    print(f"\n===== {tag} {extra} =====")
    print(f"{'B':>7} {'Me':>7} {'unf ms':>9} {'fus ms':>9} {'unf Mtok/s':>10} {'fus Mtok/s':>10} {'fus/unf':>8}")
    for r in rows:
        fm = f"{r['fus_ms']:.3f}" if r["fus_ms"] else "INFEAS"
        ft = f"{r['fus_tput']/1e6:.4f}" if r["fus_tput"] else "--"
        sp = f"{r['speedup']:.3f}" if r["speedup"] else "--"
        print(f"{r['B']:>7} {r['Me']:>7} {r['unf_ms']:>9.3f} {fm:>9} {r['unf_tput']/1e6:>10.4f} {ft:>10} {sp:>8}")
    ub = ubest["unf_tput"] / 1e6
    print(f"  UNFUSED best: B={ubest['B']} -> {ub:.4f} Mtok/s")
    if fbest:
        fbv = fbest["fus_tput"] / 1e6
        print(f"  FUSED   best: B={fbest['B']} (Me={fbest['Me']}) -> {fbv:.4f} Mtok/s")
        print(f"  >>> fused_best / unfused_best = {fbv/ub:.4f}x   "
              f"{'FUSED WINS' if fbv > ub else 'unfused wins'}")
    else:
        print("  FUSED: infeasible at every batch")
    return {
        "tag": tag, "extra": extra,
        "unfused_best": {"B": ubest["B"], "Mtok_s": ubest["unf_tput"] / 1e6},
        "fused_best": ({"B": fbest["B"], "Me": fbest["Me"], "Mtok_s": fbest["fus_tput"] / 1e6}
                       if fbest else None),
        "ratio_fused_over_unfused": (fbest["fus_tput"] / ubest["unf_tput"]) if fbest else None,
        "fused_wins": bool(fbest and fbest["fus_tput"] > ubest["unf_tput"]),
        "rows": rows,
    }


results = {}

# --------------------------------------------------------------------------- #
# (a) INTERMEDIATE width sweep (MoE baseline: experts=256, top_k=8)             #
# --------------------------------------------------------------------------- #
print("#" * 70)
print("# PART (a)  INTERMEDIATE width sweep  (MoE 256 experts, top_k 8, H100)")
print("#" * 70)
part_a = []
for INT in (512, 1024, 2048, 4096):
    configure(INT, experts=256, topk=8)
    raw, pw = m0max_pow2(INT, H100.smem_per_block_bytes)
    rows, ub, fb = run_sweep(H100)
    r = show(f"INT={INT}", rows, ub, fb, extra=f"[m0max_raw={raw} m0max_pow2={pw} -> single-pass B<= {pw*32}]")
    part_a.append(r)
results["part_a_intermediate"] = part_a

# --------------------------------------------------------------------------- #
# (b) SMEM per block x{1,2,4} on H100  (baseline INT=2048 MoE)                  #
# --------------------------------------------------------------------------- #
print("\n" + "#" * 70)
print("# PART (b)  SMEM per block x{1,2,4}  (MoE INT=2048, 256 experts, H100)")
print("#" * 70)
configure(2048, experts=256, topk=8)
part_b = []
for mult in (1, 2, 4):
    gpu = dataclasses.replace(H100, smem_per_block_bytes=int(H100.smem_per_block_bytes * mult))
    raw, pw = m0max_pow2(2048, gpu.smem_per_block_bytes)
    rows, ub, fb = run_sweep(gpu)
    r = show(f"SMEM x{mult}", rows, ub, fb,
             extra=f"[{gpu.smem_per_block_bytes//1024}KiB  m0max_raw={raw} pow2={pw} -> single-pass B<= {pw*32}]")
    part_b.append({**r, "smem_mult": mult, "smem_KiB": gpu.smem_per_block_bytes // 1024})
results["part_b_smem"] = part_b

# --------------------------------------------------------------------------- #
# (c) DENSE FFN: experts=1, top_k=1 -> Me=B  (INT=2048, H100)                   #
# --------------------------------------------------------------------------- #
print("\n" + "#" * 70)
print("# PART (c)  DENSE FFN (experts=1, top_k=1, Me=B, INT=2048, H100)")
print("#" * 70)
configure(2048, experts=1, topk=1)
# for dense, m0 can be any divisor of B (not just power-of-2), so add odd/large batches too
dense_batches = sorted(set(BATCHES + [192, 384, 768, 1536, 3072, 6144, 12288]))
rows, ub, fb = run_sweep(H100, dense_batches)
r_dense = show("DENSE experts=1", rows, ub, fb, extra="[Me=B]")
results["part_c_dense"] = r_dense

# Also dense with small INT (best chance for fused): INT=512 dense
print("\n--- (c-bonus) DENSE + small INT=512 (most favorable single-pass regime) ---")
configure(512, experts=1, topk=1)
rows2, ub2, fb2 = run_sweep(H100, dense_batches)
r_dense512 = show("DENSE experts=1 INT=512", rows2, ub2, fb2, extra="[Me=B]")
results["part_c_dense_int512"] = r_dense512

# --------------------------------------------------------------------------- #
# Verdict summary                                                              #
# --------------------------------------------------------------------------- #
print("\n" + "#" * 70)
print("# VERDICT SUMMARY  (fused_best / unfused_best;  >1 => fused wins)")
print("#" * 70)
def line(label, r):
    if r["fused_best"] is None:
        print(f"  {label:<34} fused INFEASIBLE")
    else:
        print(f"  {label:<34} {r['ratio_fused_over_unfused']:.4f}x   "
              f"{'FUSED WINS' if r['fused_wins'] else 'unfused wins'}")
for r in part_a:
    line(f"(a) {r['tag']} MoE", r)
for r in part_b:
    line(f"(b) {r['tag']} MoE INT2048", r)
line("(c) DENSE INT2048", r_dense)
line("(c) DENSE INT512", r_dense512)

any_win = (any(r["fused_wins"] for r in part_a) or any(r["fused_wins"] for r in part_b)
           or r_dense["fused_wins"] or r_dense512["fused_wins"])
print(f"\n  ANY regime with fused_best > unfused_best?  {'YES' if any_win else 'NO'}")

json.dump(results, open("exp2_fused_beats_unfused.json", "w"), indent=1)
print("[wrote exp2_fused_beats_unfused.json]")
