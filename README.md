# GEMM-mapping — latency-aware snowcat-roofline model + LLM-fusion study

Analytical GPU kernel-time model (extended from a GEMM roofline to LLM-inference
fusions) plus the physical-hardware validation of its predictions. Reorganized
2026-07-24 into the layout below.

## Core model (repository root — imported by everything)

| file | role |
|---|---|
| `gemm_time_estimator.py` | latency-aware snowcat-roofline: `GpuModel`, the `GPUS` profile registry, `estimate_gemm_time`, L2/pipeline/Little's-law model |
| `gemm_time_estimator_min.py` | trimmed variant |
| `fusion_time_estimator.py` | non-GEMM adaptation: `estimate_vector_kernel`, `estimate_fused_gemm`+`Epilogue`, `estimate_gemm_grouped`, `estimate_ffn_fused`; F1–F6 fusion specs |
| `fusion_configs.py`, `fusion_configs_prefill.py` | whole-layer fusion-config enumeration + throughput-vs-batch/token sweeps |
| `metax_c500_model.py` | MetaX C500 `GpuModel` registration |
| `snowcat_demo/` | base snowcat/Orojenesis model (tiling → traffic → roofline) + the web demo |

## Subdirectories

```
experiments/            analytical studies on the model (chain/multi-GEMM fusion, exp*, batch sweep)
validation/
  rtx4060/              RTX 4060 physical measurement harness (needs torch+triton+GPU)
  metax/                MetaX C500 physical measurement harness
  calibration/          single-GEMM est-vs-measured, L2 calibration, occupancy-BW plot, H100 compare
  microbench/           native CUDA microbenchmarks: gemm.cu (CUTLASS-vs-cuBLAS), membench, occupancy_bw
                        + build.sh/run.sh/inspect_cublas.sh  (see its own README.md)
charts/                 build_chart[_prefill].py — HTML throughput artifacts
docs/
  sim_real_synthesis.md the integrated H100/C500/4060 est-vs-real conclusion
  results/              *_results.md / *_table.md (generated result write-ups)
  tasks/                RTX4060_*_TASK.md (remote-machine task specs)
data/
  json/                 all result JSONs (estimator outputs + measured runs)
  device/               device dumps: *_smi.txt, occ_bw.csv, occupancy_bw.png, estimator_validation.csv
  charts/               generated *_throughput.html
notes/                  working notes / logs
result/                 standalone report artifact
```

## Running

**Run scripts from this directory (the GEMM-mapping root)** — data I/O paths are
relative to it (e.g. `data/json/…`). A path-shim at the top of every script under
`experiments/`, `validation/*`, `charts/` walks up to this root so their
`import gemm_time_estimator` (and the other core modules) resolve regardless of cwd.

```bash
# analytical estimator (pure Python, no GPU):
python3 fusion_time_estimator.py --gpu both --verbose
python3 experiments/exp2_verify.py
python3 fusion_configs.py --gpu h100-sxm            # writes data/json/fusion_configs.json
python3 charts/build_chart.py                        # reads data/json/, writes data/charts/

# physical validation (needs a GPU + torch/triton; pass --out data/json/...):
python3 validation/rtx4060/rtx4060_residual_down.py --out data/json/rtx4060_residual_down.json
```

The `validation/rtx4060/*` and `validation/metax/*` scripts import `torch`/`triton`
and only run on their respective hardware; the estimator and `experiments/` run anywhere.

## Where to start

New here? Read `docs/sim_real_synthesis.md` for the conclusions, then
`fusion_time_estimator.py` for how non-GEMM kernels are modeled. (A full developer
guide — concept → API → extension → validation — was drafted as a repo-root `SKILL.md`.)
