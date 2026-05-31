import argparse
import os
import statistics
import time
from contextlib import contextmanager

import torch

from kernels.cpu import gpt as cpu_gpt
from kernels.cpu.providers import select_provider
from kernels.refs import gpt2 as ref_gpt


def _torch_native_rope(x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    x2 = x.reshape(x.shape[0], x.shape[1] // 2, 2)
    cs2 = cos_sin.reshape(cos_sin.shape[0], cos_sin.shape[1] // 2, 2)
    x0 = x2[..., 0]
    x1 = x2[..., 1]
    cos = cs2[..., 0]
    sin = cs2[..., 1]
    out = torch.empty_like(x)
    out2 = out.reshape(x.shape[0], x.shape[1] // 2, 2)
    out2[..., 0] = x0 * cos + x1 * sin
    out2[..., 1] = x0 * (-sin) + x1 * cos
    return out


def _torch_native_gemm_rmsnorm(A: torch.Tensor, B: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    return (torch.mm(A, B) * R.reshape(-1, 1)).to(dtype=A.dtype)


def _torch_native_gemm_rmsnorm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    D = torch.mm(A, B) * R.reshape(-1, 1)
    D2 = D.reshape(D.shape[0], D.shape[1] // 2, 2)
    O = torch.nn.functional.silu(D2[..., 0]) * D2[..., 1]
    return D.to(dtype=A.dtype), O.to(dtype=A.dtype)


def _torch_native_gemm_rmsnorm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    cos_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    D = torch.mm(A, B) * R.reshape(-1, 1)
    O = _torch_native_rope(D, cos_sin)
    return D.to(dtype=A.dtype), O.to(dtype=A.dtype)


def _bench(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def _time_once(fn) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000.0


@contextmanager
def _provider_env(provider: str):
    previous = os.environ.get("CODA_CPU_PROVIDER")
    os.environ["CODA_CPU_PROVIDER"] = provider
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CODA_CPU_PROVIDER", None)
        else:
            os.environ["CODA_CPU_PROVIDER"] = previous


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

    torch_cases = {
        "torch-native/gemm_rmsnorm": lambda: _torch_native_gemm_rmsnorm(A, B, R),
        "torch-native/gemm_rmsnorm_swiglu": lambda: _torch_native_gemm_rmsnorm_swiglu(A, B, R),
        "torch-native/gemm_rmsnorm_rope": lambda: _torch_native_gemm_rmsnorm_rope(A, B, R, cos_sin),
        "torch-compiled/gemm_rmsnorm": lambda: ref_gpt.gemm_rmsnorm(A, B, R),
        "torch-compiled/gemm_rmsnorm_swiglu": lambda: ref_gpt.gemm_rmsnorm_swiglu(A, B, R),
        "torch-compiled/gemm_rmsnorm_rope": lambda: ref_gpt.gemm_rmsnorm_rope(A, B, R, cos_sin),
    }
    for name, fn in torch_cases.items():
        cold_ms = _time_once(fn)
        median_ms = _bench(fn, warmup=args.warmup, repeats=args.repeats)
        print(f"{name}: cold={cold_ms:.3f} ms warm_median={median_ms:.3f} ms")

    provider_cases = {
        "gemm_rmsnorm": lambda: cpu_gpt.gemm_rmsnorm(A, B, R),
        "gemm_rmsnorm_swiglu": lambda: cpu_gpt.gemm_rmsnorm_swiglu(A, B, R),
        "gemm_rmsnorm_rope": lambda: cpu_gpt.gemm_rmsnorm_rope(A, B, R, cos_sin),
    }
    for provider in ("aten", "onednn-x64-brgemm", "libxsmm"):
        try:
            select_provider(provider)
        except RuntimeError as exc:
            print(f"{provider}: skipped ({exc})")
            continue

        with _provider_env(provider):
            for case_name, fn in provider_cases.items():
                cold_ms = _time_once(fn)
                median_ms = _bench(fn, warmup=args.warmup, repeats=args.repeats)
                print(f"{provider}/{case_name}: cold={cold_ms:.3f} ms warm_median={median_ms:.3f} ms")


if __name__ == "__main__":
    main()
