"""Compare snowcat-roofline estimates (C500 model) vs measured C500 times. Run in 'area' env."""
import json, statistics, math
import metax_c500_model  # registers 'metax-c500'
from gemm_time_estimator import GPUS, optimal_mapping_by_time

G = GPUS["metax-c500"]
D = json.load(open("metax_measured.json"))


def est(m, n, k):
    try:
        _, e = optimal_mapping_by_time(m, n, k, G, l2=True)
        return e.time_s
    except Exception as ex:
        return None


print(f"# Estimator (C500 model) vs measured — {D['device']}  (ratio = est/meas)\n")
ratios = []
print(f"{'group':>12} {'M':>7} {'N':>6} {'K':>6} {'meas ms':>9} {'est ms':>9} {'est/meas':>9} {'meas TF':>8}")
for group, rows in D["gemms"].items():
    seen = set()
    for r in rows:
        key = (r["m"], r["n"], r["k"])
        if key in seen:
            continue
        seen.add(key)
        e = est(*key)
        meas = r["t_s"]
        if e is None:
            print(f"{group:>12} {r['m']:>7} {r['n']:>6} {r['k']:>6} {meas*1e3:>9.4f} {'n/a':>9} {'--':>9} {r['tflops']:>8.1f}")
            continue
        ratio = e / meas
        ratios.append(ratio)
        print(f"{group:>12} {r['m']:>7} {r['n']:>6} {r['k']:>6} {meas*1e3:>9.4f} {e*1e3:>9.4f} {ratio:>9.3f} {r['tflops']:>8.1f}")

print(f"\n## Single-GEMM accuracy over {len(ratios)} GEMMs (est/meas):")
gm = math.exp(statistics.mean(math.log(x) for x in ratios))
print(f"  median {statistics.median(ratios):.3f}   geomean {gm:.3f}   "
      f"min {min(ratios):.3f}   max {max(ratios):.3f}")
within2 = sum(1 for x in ratios if 0.5 <= x <= 2.0)
within15 = sum(1 for x in ratios if 1 / 1.5 <= x <= 1.5)
print(f"  within 1.5x: {within15}/{len(ratios)} ({100*within15/len(ratios):.0f}%)   "
      f"within 2x: {within2}/{len(ratios)} ({100*within2/len(ratios):.0f}%)")

print(f"\n## Unfused chains: measured vs estimator (sum of per-GEMM estimates):")
print(f"{'chain':>26} {'meas ms':>9} {'est ms':>9} {'est/meas':>9}")
for name, c in D["chains"].items():
    widths = c["widths"]; M = c["M"]
    gemms = [(M, widths[s + 1], widths[s]) for s in range(c["L"])]
    es = [est(*g) for g in gemms]
    if any(e is None for e in es):
        print(f"{name:>26} {c['t_s']*1e3:>9.4f} {'n/a':>9} {'--':>9}")
        continue
    etot = sum(es)
    print(f"{name:>26} {c['t_s']*1e3:>9.4f} {etot*1e3:>9.4f} {etot/c['t_s']:>9.3f}")
