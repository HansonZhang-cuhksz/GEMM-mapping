"""Focused 2-GEMM chain sweep in the shape regime where fusion is actually balanced.

The 27-shape sweep (chain_gemm_fusion.py) forced every dim >= 2048, so the fused kernel's
held intermediate slice (min(N1,N2) wide) never fit a decent row-block and fusion almost
always lost. Here we deliberately put the shapes in the *flash-attention regime*, where the
fusion tradeoff is tunable:

  * NARROW output N2 (128/256) -> the fused kernel holds a small E-accumulator E[m0,N2] and
    streams the wide intermediate -> feasible with a large row-block m0 (mt = M/m0 small).
  * WIDE intermediate N1 (4096/8192) -> C = M*N1 is large (> 30 MB eff-L2) so the UNFUSED
    pair pays a real HBM round-trip that fusion avoids -> a saving worth chasing.
  * K1 = the balance knob -> weight B = K1*N1 sweeps from < eff-L2 (re-read from L2, ~free ->
    fusion wins) to > eff-L2 (re-read mt x from DRAM -> fusion loses). D = N1*N2 stays small.

M is fixed (batch). Everything is timed by the SAME snowcat-roofline estimator via the
unfused_time / fused_time functions from chain_gemm_fusion.py (no new physics here).

Run:  conda run -n area python chain_gemm_focused.py            # terminal table
      conda run -n area python chain_gemm_focused.py --out chain_gemm_focused_table.md
"""

from __future__ import annotations

import argparse
import itertools
import math

from gemm_time_estimator import GPUS, GpuModel
from chain_gemm_fusion import BPE, _eff_l2, fused_time, unfused_time

M = 8192                              # batch (fixed): C = M*N1 is large; tile-unlimited
N1_VALUES = (4096, 8192)             # intermediate width -> big C, and lets B spill L2
N2_VALUES = (128, 256)              # narrow output -> hold small E-accumulator (feasible)
K1_VALUES = (512, 1024, 2048, 4096, 8192, 16384)   # the balance knob (B = K1*N1)


def _fmt_mib(b: float) -> str:
    return f"{b/2**20:.1f}"


def run(gpu: GpuModel, out_path: str | None = None) -> None:
    eff_l2 = _eff_l2(gpu)
    header = (
        f"# 2-GEMM chain fusion — focused (flash-attention regime), {gpu.name}\n\n"
        f"Chain `C[M,N1]=A[M,K1]@B[K1,N1]`, `E[M,N2]=C[M,N1]@D[N1,N2]`. **M={M} fixed.** "
        f"Narrow output N2 {list(N2_VALUES)}, wide intermediate N1 {list(N1_VALUES)}, "
        f"K1 (the balance knob) {list(K1_VALUES)}. All times from the snowcat-roofline "
        f"estimator. eff-L2 = {eff_l2/2**20:.0f} MiB; SMEM/blk {gpu.smem_per_block_bytes//1024} KiB.\n\n"
        f"C = M·N1·2 is {M*N1_VALUES[0]*BPE/2**20:.0f}–{M*N1_VALUES[-1]*BPE/2**20:.0f} MiB "
        f"(always > eff-L2 → unfused pays an HBM round-trip). B = K1·N1·2 crosses eff-L2 as K1 grows.\n\n"
        "| N1 | N2 | K1 | B (MiB) | B loc | C (MiB) | m0×mt | unfused ms | fused ms | speedup | winner |\n"
        "|---:|---:|---:|---:|:--:|---:|---|---:|---:|---:|:--:|"
    )
    lines = [header]
    print(header)
    nf = nu = ni = 0
    for N1, N2 in itertools.product(N1_VALUES, N2_VALUES):
        for K1 in K1_VALUES:
            u_t, ui = unfused_time(M, K1, N1, N2, gpu)
            f_t, fi = fused_time(M, K1, N1, N2, gpu)
            b_bytes = K1 * N1 * BPE
            b_loc = "L2" if b_bytes <= eff_l2 else "HBM"
            c_mib = _fmt_mib(M * N1 * BPE)
            if not math.isfinite(f_t):
                m0mt, fms, sp, win = "INFEAS", "—", "—", "infeasible"
                ni += 1
            else:
                m0mt = f"{fi['m0']}×{fi['mt']}"
                fms = f"{f_t*1e3:.3f}"
                sp = f"{u_t/f_t:.3f}×"
                if f_t < u_t:
                    win, nf = "**FUSE**", nf + 1
                else:
                    win, nu = "unfuse", nu + 1
            row = (f"| {N1} | {N2} | {K1} | {_fmt_mib(b_bytes)} | {b_loc} | {c_mib} | {m0mt} | "
                   f"{u_t*1e3:.3f} | {fms} | {sp} | {win} |")
            lines.append(row)
            print(row)
        lines.append("| | | | | | | | | | | |")   # visual break between (N1,N2) groups
    total = (f"\n**Totals: FUSE {nf} / unfuse {nu} / infeasible {ni}** "
             f"(of {len(N1_VALUES)*len(N2_VALUES)*len(K1_VALUES)}).")
    lines.append(total)
    print(total)
    if out_path:
        with open(out_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\n[wrote {out_path}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", choices=sorted(GPUS), default="h100-sxm")
    ap.add_argument("--out", default=None, help="also write the markdown table to this file")
    args = ap.parse_args()
    run(GPUS[args.gpu], args.out)


if __name__ == "__main__":
    main()
