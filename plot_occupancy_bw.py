#!/usr/bin/env python
"""Plot measured effective bandwidth vs occupancy against the bw*sm_util model.

Reads occ_bw.csv (from ./occupancy_bw). Saves occupancy_bw.png.
"""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette (canonical, validated in the literature).
C_MEAS = "#0072B2"   # blue   -- measured
C_LIN  = "#D55E00"   # vermillion -- current model bw*sm_util (linear)
C_SAT  = "#009E73"   # green  -- proposed roofline: min(peak, per_unit*active)
GRID   = "#DADCE0"; INK = "#202124"; MUTED = "#5F6368"

rows = list(csv.DictReader(open("occ_bw.csv")))
def series(name):
    xs = [float(r["x_occupancy"]) for r in rows if r["sweep"] == name]
    ys = [float(r["bw_GBs"])      for r in rows if r["sweep"] == name]
    return xs, ys

sm_x, sm_y = series("smcount")
wp_x, wp_y = series("warps")
PEAK = max(sm_y)                       # measured plateau
per_sm = sm_y[0] / sm_x[0]             # GB/s per unit of occupancy in the linear ramp

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=130)

# ---- Panel A: SM-count occupancy (== the model's sm_util), first wave 0..1 ----
a_x = [x for x in sm_x if x <= 1.001]; a_y = sm_y[:len(a_x)]
axA.plot(a_x, a_y, "-o", color=C_MEAS, lw=2, ms=5, label="measured bw_eff", zorder=5)
axA.plot([0, 1], [0, PEAK], "--", color=C_LIN, lw=2, label="model: bw·sm_util (linear)")
sat = [min(PEAK, per_sm * x) for x in a_x]
axA.plot(a_x, sat, "--", color=C_SAT, lw=2, label="roofline: min(peak, rate·active)")
knee_occ = PEAK / per_sm          # occupancy at which the ramp hits the plateau
knee_sms = knee_occ * 24
axA.axhline(PEAK, color=GRID, lw=1, zorder=0)
axA.annotate(f"plateau ≈ {PEAK:.0f} GB/s\nreached by ~{knee_occ*100:.0f}% occ (~{knee_sms:.0f} SMs)",
             xy=(0.40, PEAK), xytext=(0.44, PEAK*0.52), color=MUTED, fontsize=9,
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
axA.annotate(f"at 33% occ:\nmodel {PEAK*0.333:.0f}, real ~246 GB/s", xy=(0.333, 246),
             xytext=(0.36, 95), color=INK, fontsize=9,
             arrowprops=dict(arrowstyle="->", color=INK, lw=1))
axA.set_title("(A) effective BW vs SM-count occupancy", color=INK)
axA.set_xlabel("occupancy  =  active SMs / 24   (the model's sm_util)")
axA.set_ylabel("effective bandwidth (GB/s)")
axA.set_xlim(0, 1.02); axA.set_ylim(0, PEAK*1.08)
axA.legend(frameon=False, fontsize=9, loc="lower right")
axA.grid(True, color=GRID, lw=0.6); axA.set_axisbelow(True)
for s in ("top", "right"): axA.spines[s].set_visible(False)

# ---- Panel B: warps-per-SM occupancy (all 24 SMs active) ----
axB.plot(wp_x, wp_y, "-o", color=C_MEAS, lw=2, ms=5, label="measured bw_eff", zorder=5)
axB.plot([0, 1], [0, PEAK], "--", color=C_LIN, lw=2, label="linear in per-SM occupancy")
axB.axhline(PEAK, color=GRID, lw=1, zorder=0)
axB.annotate("saturates by ~25% warp occupancy\n(≈8 warps/SM)", xy=(0.25, 244),
             xytext=(0.30, 110), color=MUTED, fontsize=9,
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
axB.set_title("(B) effective BW vs warps-per-SM occupancy", color=INK)
axB.set_xlabel("occupancy  =  threads/block / 1536   (all 24 SMs active)")
axB.set_ylabel("effective bandwidth (GB/s)")
axB.set_xlim(0, 0.70); axB.set_ylim(0, PEAK*1.08)
axB.legend(frameon=False, fontsize=9, loc="lower right")
axB.grid(True, color=GRID, lw=0.6); axB.set_axisbelow(True)
for s in ("top", "right"): axB.spines[s].set_visible(False)

fig.suptitle("Effective DRAM bandwidth saturates with occupancy — it is NOT bw·sm_util  "
             "(RTX 4060 Laptop, 512 MiB read stream)", fontsize=11, color=INK, y=1.02)
fig.tight_layout()
fig.savefig("occupancy_bw.png", bbox_inches="tight", facecolor="white")
print(f"peak={PEAK:.1f} GB/s  1-SM BW={sm_y[0]:.1f} GB/s  knee~{knee_occ*100:.0f}% occ "
      f"(~{knee_sms:.1f} SMs)  -> saved occupancy_bw.png")
