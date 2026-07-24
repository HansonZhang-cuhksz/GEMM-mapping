**All real results are benchmarked on RTX4060, 1500MHz core, 5501MHz VRAM.**
**"Estimation Model stands for Latency-Aware Snowcat-Roofline Model."**

# Chain GEMM
C=A@B, E=C@D, G=E@F, ...
## Relation
### Shape
**Conclusion：The fusion optimality of chain GEMM is not related to shape of the GEMMs**

### Size
**Conclusion：The fusion optimality of chain GEMM is related to size of the GEMMs**

# Dense LLM (GLM 5.2 Parameter Set)
MLA, Residual Add, RMSNorm, Up_Gate, SwiGLU, Down, Residual Add

**Most Optimal Fusion in most GPUs:**
1. MLA + Residual Add + RMSNorm
2. Up_Gate + SwiGLU
3. Down + Residual Add

## MLA + Residual Add + RMSNorm (Optimal)
Only benchmarked on H100-SXM estimation model.

| batch B | mla_o  |  res1  | rmsnorm | UNFUSED | FUSED (S3) |  gain  |
|---------|--------|--------|---------|---------|------------|--------|
|     512 | 0.1269 | 0.0056 |  0.0019 |  0.1345 |     0.1288 | 1.0437 |
|    1024 | 0.2539 | 0.0113 |  0.0038 |  0.2689 |     0.2576 | 1.0437 |
|    2048 | 0.5077 | 0.0225 |  0.0075 |  0.5377 |     0.5152 | 1.0437 |
|    4096 | 1.0153 | 0.0451 |  0.0150 |  1.0754 |     1.0303 | 1.0437 |
|    8192 | 2.0304 | 0.0901 |  0.0301 |  2.1506 |     2.0605 | 1.0437 |
|   16384 | 4.0608 | 0.1803 |  0.0601 |  4.3012 |     4.1209 | 1.0438 |

By estimation model, MLA + Residual Add + RMSNorm fusion can consistently provide ~4.37% throughput improvement on standard datacenter GPU. The improvement is irrelative with batch size.

## MLA + Residual Add (Optimal)
RTX4060's SMEM is not sufficient to fuse full `MLA + Residual Add + RMSNorm`. Under this constraint, `MLA + Residual Add` is the most optimal fusion.

### H100-SXM Estimation
| batch B | mla_o  |  res1  | UNFUSED | FUSED (S2) |  gain  |
|---------|--------|--------|---------|------------|--------|
|     512 | 0.1269 | 0.0056 |  0.1326 |     0.1288 | 1.0292 |
|    1024 | 0.2539 | 0.0113 |  0.2652 |     0.2576 | 1.0292 |
|    2048 | 0.5077 | 0.0225 |  0.5302 |     0.5152 | 1.0292 |
|    4096 | 1.0153 | 0.0451 |  1.0603 |     1.0303 | 1.0292 |
|    8192 | 2.0304 | 0.0901 |  2.1206 |     2.0605 | 1.0292 |
|   16384 | 4.0608 | 0.1803 |  4.2411 |     4.1209 | 1.0292 |

### RTX4060 Estimation
| batch B |  mla_o   |  res1  | UNFUSED  | FUSED (S2) |  gain  |
|---------|----------|--------|----------|------------|--------|
|     512 |   5.5924 | 0.1110 |   5.7034 |     5.5924 | 1.0199 |
|    1024 |  11.1848 | 0.2221 |  11.4069 |    11.1848 | 1.0199 |
|    2048 |  22.3696 | 0.4441 |  22.8137 |    22.3696 | 1.0199 |
|    4096 |  44.7392 | 0.8882 |  45.6274 |    44.7392 | 1.0199 |
|    8192 |  89.4785 | 1.7764 |  91.2549 |    89.4785 | 1.0199 |
|   16384 | 178.9570 | 3.5528 | 182.5098 |   178.9570 | 1.0199 |

### RTX4060 Real
| batch B |  mla_o   |  res1  | UNFUSED  | FUSED (S2) |  gain  |
|---------|----------|--------|----------|------------|--------|
|     512 |   5.5924 | 0.1110 |   5.7034 |     5.5924 | 1.0199 |
|    1024 |  11.1848 | 0.2221 |  11.4069 |    11.1848 | 1.0199 |
|    2048 |  22.3696 | 0.4441 |  22.8137 |    22.3696 | 1.0199 |
|    4096 |  44.7392 | 0.8882 |  45.6274 |    44.7392 | 1.0199 |
|    8192 |  89.4785 | 1.7764 |  91.2549 |    89.4785 | 1.0199 |
|   16384 | 178.9570 | 3.5528 | 182.5098 |   178.9570 | 1.0199 |

## RMSNorm + Up_Gate (Suboptimal)
Prologue-GEMM fusion

|Method|Unfused|Fused|Gain|
|-|-|-|-|
|Real|0.4361|0.5535|0.743|
|Estimation RTX4060|||1.024|
|Estimation H100||||

## Up_Gate + SwiGLU (Optimal)
Using CODA-style GEMM-epilogue fusion.

|Method|Unfused|Fused|Gain|
|-|-|-|-|
|Real cuBLAS|8.628|7.933|1.088|
|Real Triton Custom Kernel|8.952|7.796|1.148|
|Estimation RTX4060|8.063|7.471|1.079|
|Estimation H100||||

## Down + Residual Add (Optimal)

### Dense 4*H stock addmm
|Method|Unfused|Fused|Gain|
|-|-|-|-|
|Real cuBLAS|179.647|179.207|1.003|
|Real Triton Custom Kernel|255.850|253.575|1.009|
|Estimation RTX4060|135.994|134.218|1.013|
|Estimation H100||||

### Narrow K=H stock addmm
|Method|Unfused|Fused|Gain|
|-|-|-|-|
|Real cuBLAS|6.594|6.276|1.051|
|Real Triton Custom Kernel|6.738|5.843|1.153|
|Estimation RTX4060|||1.159|
|Estimation H100||||

# MoE LLM (GLM 5.2 Parameter Set)
MLA, Residual Add, RMSNorm, Router, top-k, Up_Gate, SwiGLU, Down, Expert Merge, Residual Add

**Most Optimal Fusion in most GPUs:**
1. MLA + Residual Add + RMSNorm
2. Up_Gate + SwiGLU
3. Down + Expert Merge + Residual Add

## Residual Add + Expert Merge (Optimal)
|Method|Unfused|Fused|Gain|
|-|-|-|-|
|Real|7.634|6.145|1.242|
|Estimation RTX4060|7.106|5.921|1.200|
|Estimation H100||||