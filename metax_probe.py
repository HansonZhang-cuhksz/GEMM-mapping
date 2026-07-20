"""Probe MetaX C500: peak BF16 TFLOP/s, HBM bandwidth, sanity GEMM timings.

Run in the 'fusion' env (torch 2.8 + metax). Proper timing: warmup + cuda Events + median.
"""
import torch, time, statistics

DEV = "cuda:0"
torch.backends.cuda.matmul.allow_tf32 = True


def sync():
    torch.cuda.synchronize()


def time_gemm(m, n, k, dtype=torch.bfloat16, iters=50, warmup=15):
    a = torch.randn(m, k, device=DEV, dtype=dtype)
    b = torch.randn(k, n, device=DEV, dtype=dtype)
    for _ in range(warmup):
        c = a @ b
    sync()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); c = a @ b; e.record(); sync()
        ts.append(s.elapsed_time(e) / 1e3)  # s
    t = statistics.median(ts)
    tflops = 2 * m * n * k / t / 1e12
    del a, b, c; torch.cuda.empty_cache()
    return t, tflops


def time_bw(nbytes=1 << 30, iters=50, warmup=15):
    n = nbytes // 4  # fp32 elems
    x = torch.randn(n, device=DEV, dtype=torch.float32)
    y = torch.empty_like(x)
    for _ in range(warmup):
        y.copy_(x)
    sync()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); y.copy_(x); e.record(); sync()
        ts.append(s.elapsed_time(e) / 1e3)
    t = statistics.median(ts)
    gbps = 2 * nbytes / t / 1e9  # read + write
    del x, y; torch.cuda.empty_cache()
    return t, gbps


if __name__ == "__main__":
    p = torch.cuda.get_device_properties(0)
    print(f"# {p.name}: {p.multi_processor_count} SM, L2={p.L2_cache_size/2**20:.0f}MiB, "
          f"SMEM/blk={p.shared_memory_per_block/1024:.0f}KiB, mem={p.total_memory/2**30:.0f}GiB")

    print("\n## Peak BF16 GEMM TFLOP/s (large square, compute-bound):")
    best = 0
    for n in [4096, 8192, 12288, 16384]:
        t, tf = time_gemm(n, n, n)
        best = max(best, tf)
        print(f"  {n:>5}^3: {t*1e3:8.3f} ms  {tf:7.1f} TFLOP/s")
    print(f"  => peak ~ {best:.0f} TFLOP/s (bf16)")

    print("\n## HBM bandwidth (fp32 copy, read+write):")
    bestbw = 0
    for mb in [256, 512, 1024, 2048]:
        t, gb = time_bw(mb << 20)
        bestbw = max(bestbw, gb)
        print(f"  {mb:>5} MiB: {t*1e3:8.3f} ms  {gb:7.0f} GB/s")
    print(f"  => peak BW ~ {bestbw:.0f} GB/s = {bestbw/1e3:.2f} TB/s")

    print("\n## Sanity: a few study GEMMs (bf16):")
    for (m, n, k, tag) in [(2048, 6144, 16384, "mla_o decode"), (8192, 8192, 8192, "square 8k"),
                            (131072, 128, 128, "tall-skinny chain stage")]:
        t, tf = time_gemm(m, n, k, iters=30, warmup=10)
        print(f"  [{m},{k}]@[{k},{n}] {tag:>24}: {t*1e3:8.3f} ms  {tf:7.1f} TFLOP/s")
