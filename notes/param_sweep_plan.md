# Does fusion benefit depend on model parameters? — plan & log

Question: GLM-5.2 decode fusions gave ~0 benefit on C500. Is that intrinsic to the algorithmic
PATTERN (its specific params) or would other model parameters make fusion worthwhile?

## Hypothesis (from the roofline physics)
Fusion folds a memory-bound vector op (traffic V) into a GEMM. Benefit fraction:
 - vs a COMPUTE-bound GEMM: (V/BW)/(compute) ~ ridge/dim  -> SMALLER models fuse better; independent
   of token count; ~ridge/dim = 158/6144 = 2.6% for GLM on C500.
 - vs a WEIGHT-bound MoE GEMM: V/weight_bytes ~ tokens/(experts*dim) -> negligible for many experts /
   large dim (GLM: 16384/(256*2048), tiny). Fewer experts / dense / smaller dim -> bigger.
Prediction: fusion is worthwhile only for SMALL and/or DENSE models (few experts), where the GEMMs are
small enough that the activation/vector traffic is a meaningful fraction. GLM-5.2's large 256-expert
MoE is close to the worst case.

## Experiment (measure on physical C500, fusion env)
Fusion "ceiling" per config = vec_kernel_ms / (gemm_ms + vec_kernel_ms) (best case: vector fully
absorbed), + bound classification (compute if achieved TF/s > 0.7*226 else memory).
 - Axis 1 DIM: dense FFN up_gate+SwiGLU, M=2048, sweep hidden {1024,2048,4096,8192} -> ceiling ~ ridge/dim?
 - Axis 2 MoE-vs-DENSE: fixed 16384 tokens, hidden 2048, experts {1,8,64,256} -> tokens/expert varies.
 - Axis 3 ATTENTION F1: mla_o [batch,6144,16384] + residual, batch {256,1024,4096,16384}.
Save param_sweep.json; then workflow to analyze + verify + synthesize where fusion pays off.

## Steps
1. [x] Build + run metax_param_sweep.py on C500.
2. [ ] Workflow: analyze the ceiling vs params, verify, synthesize "which params make fusion worth it".

## Results (C500, fusion ceiling = vec/(gemm+vec))
Axis 1 DIM (dense FFN+SwiGLU, M=2048): H=1024 -> 28.3% (memory), 2048 -> 19.9%, 4096 -> 10.6%,
  8192 -> 5.4%. => ceiling ~ ridge/dim CONFIRMED: small models fuse ~5x better than large.
Axis 2 MoE (16384 tok, H=2048): E=1/8/64 (compute) -> ~16%; E=256 (m=64, weight-bound 89TF) -> 7.7%
  (but realizable < ceiling since weight-bound gemm has no spare BW; F5-style traffic saving ~3%).
Axis 3 ATTN (mla_o+residual): B=256 -> 7.3%, 1024 -> 3.7%, 4096/16384 -> 2.8%. Small batch fuses more.
=> INTERIM (CORRECTED BELOW): "small models fuse 5x better, 28% ceiling."

## CORRECTION (audit, 2 high-severity):
1. The 28% is ~67% LAUNCH-OVERHEAD artifact (sub-0.1ms vec kernels, ~22us floor). Overhead-corrected
   Axis1 ceilings -> ~11/12/8.5/4.6%; spread shrinks 5.6x -> 2.5x; H=1024 no longer leads (it's
   small/latency-bound w/ 7x spare BW, not memory-bound).
2. REALIZED benefit (torch.compile swiglu, cuBLAS addmm) is ~0 or NEGATIVE for EVERY model size:
   compile swiglu-into-GEMM -63/-23/-11% at H=512/1024/2048 (SLOWER); addmm residual -3.6%; best
   anywhere +2.2% (MoE E=256). So params move the CEILING but NOT the realized benefit on this stack.

FINAL ANSWER (half yes / half no):
 - YES GLM-5.2's ~0% is intrinsic to its params: weight-bound MoE (up_gate 97% of BW floor), ceiling
   = 2*tokens/(experts*hidden) = 2.08%. Wide hidden + 256 experts = worst case. Ridge/dim law holds
   for compute-bound H>=2048 (ceiling=41378/H, R^2=0.996).
 - NO you can't unlock fusion by changing params on the STOCK stack: cuBLAS/torch.compile can't fuse
   elementwise into (grouped) GEMM -> realized ~0% for all sizes. Higher small-dense ceilings need a
   custom CUTLASS/Triton kernel regardless of model parameters.
Results -> metax_param_results.md.
