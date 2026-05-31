import argparse
import statistics
import time

import torch

from kernels.cpu import gpt as cpu_gpt
from kernels.refs import gpt2 as ref_gpt


def _bench(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU CODA forward kernels.")
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    A = torch.randn((args.m, args.k), dtype=torch.float32)
    B = torch.randn((args.k, args.n), dtype=torch.float32)
    R = torch.rsqrt((A ** 2).mean(dim=-1) + 1e-6)
    cos = torch.ones((args.m, args.n // 2), dtype=torch.float32)
    sin = torch.zeros((args.m, args.n // 2), dtype=torch.float32)
    cos_sin = torch.stack((cos, sin), dim=-1).reshape(args.m, args.n)

    cases = {
        "cpu/gemm_rmsnorm": lambda: cpu_gpt.gemm_rmsnorm(A, B, R),
        "torch/gemm_rmsnorm": lambda: ref_gpt.gemm_rmsnorm(A, B, R),
        "cpu/gemm_rmsnorm_swiglu": lambda: cpu_gpt.gemm_rmsnorm_swiglu(A, B, R),
        "torch/gemm_rmsnorm_swiglu": lambda: ref_gpt.gemm_rmsnorm_swiglu(A, B, R),
        "cpu/gemm_rmsnorm_rope": lambda: cpu_gpt.gemm_rmsnorm_rope(A, B, R, cos_sin),
        "torch/gemm_rmsnorm_rope": lambda: ref_gpt.gemm_rmsnorm_rope(A, B, R, cos_sin),
    }

    for name, fn in cases.items():
        median_ms = _bench(fn, warmup=args.warmup, repeats=args.repeats)
        print(f"{name}: {median_ms:.3f} ms")


if __name__ == "__main__":
    main()
