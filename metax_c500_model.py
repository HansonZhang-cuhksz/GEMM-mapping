"""MetaX C500 GpuModel for the snowcat-roofline estimator, calibrated to THIS machine.

All values measured on the physical C500 (see notes/metax_c500_plan.md), except the two the
estimator shares across all profiles (l2_capacity_alpha, bw_saturation_sms) and the DRAM latency,
which are flagged as estimates. Importing this registers 'metax-c500' in gemm_time_estimator.GPUS.

Measured (fusion env, torch 2.8+metax, device 0):
  * peak BF16 GEMM      : ~226 TFLOP/s (4096..16384 cube, cuBLAS-equiv; near-flat 219-226)
  * HBM bandwidth       : ~1.43 TB/s   (fp32 copy, 2 GiB, read+write)
  * SMs                 : 104          (torch multi_processor_count)
  * L2                  : 8 MiB        (torch L2_cache_size)
  * SMEM/block & /SM    : 64 KiB       (torch shared_memory_per_block(_optin))
  * XCORE clock         : 1125 MHz     (mx-smi --show-clocks xcore, held under load)
"""
from __future__ import annotations

from dataclasses import replace
from gemm_time_estimator import GpuModel, GPUS

_PEAK_TFLOPS = 226e12
_CLOCK = 1125e6
_NUM_SM = 104

METAX_C500 = GpuModel(
    name="MetaX C500 (measured)",
    num_sm=_NUM_SM,
    tensor_cores=_NUM_SM,                                        # 1 "unit"/SM proxy; only the product matters
    tensor_flops_per_core_per_clock=_PEAK_TFLOPS / (_NUM_SM * _CLOCK),  # -> peak_tensor_flops = 226 TFLOP/s
    clock_hz=_CLOCK,
    bw_bytes_per_s=1.43e12,                                      # measured HBM copy bandwidth
    smem_per_block_bytes=64 * 1024,                             # measured (torch)
    smem_per_sm_bytes=64 * 1024,
    hbm_latency_cycles=round(400e-9 * _CLOCK),                  # ESTIMATE: ~400 ns HBM latency -> 450 cyc @1.125GHz
    bytes_per_element=2,
    l2_bytes=8 * 1024 * 1024,                                    # measured (torch)
    l2_capacity_alpha=0.6,                                       # shared estimator constant (not C500-calibrated)
    bw_saturation_sms=20.0,                                      # ASSUMPTION: ~20% of SMs (as H100 profile), uncalibrated
)

GPUS["metax-c500"] = METAX_C500

# Sanity when run directly.
if __name__ == "__main__":
    g = METAX_C500
    print(f"{g.name}: peak {g.peak_tensor_flops/1e12:.0f} TFLOP/s, BW {g.bw_bytes_per_s/1e12:.2f} TB/s, "
          f"{g.num_sm} SM, L2 {g.l2_bytes/2**20:.0f} MiB, SMEM/blk {g.smem_per_block_bytes//1024} KiB, "
          f"latency {g.latency_seconds*1e9:.0f} ns, ridge OI {g.peak_tensor_flops/g.bw_bytes_per_s:.0f} FLOP/B")
