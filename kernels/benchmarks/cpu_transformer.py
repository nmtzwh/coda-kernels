import argparse
import os
import statistics
import time
from contextlib import contextmanager

import torch

from models import ops
from kernels.cpu.providers import select_provider


def _rope_native(x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
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


def _torch_native_layer(
    x0: torch.Tensor,
    y0: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    wn0: torch.Tensor,
    wn1: torch.Tensor,
    cos_sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, hidden_dim = x0.shape
    x0_2d = x0.reshape(batch_size * seq_len, hidden_dim)
    y0_2d = y0.reshape(batch_size * seq_len, hidden_dim)

    x1 = torch.addmm(x0_2d, y0_2d, w0)
    h1 = torch.nn.functional.rms_norm(x1, normalized_shape=(hidden_dim,), weight=wn0, eps=eps)
    z1 = torch.mm(h1, w1)
    z1_2 = z1.reshape(z1.shape[0], z1.shape[1] // 2, 2)
    y1 = torch.nn.functional.silu(z1_2[..., 0]) * z1_2[..., 1]

    x2 = torch.addmm(x1, y1, w2)
    h2 = torch.nn.functional.rms_norm(x2, normalized_shape=(hidden_dim,), weight=wn1, eps=eps)
    z2 = torch.mm(h2, w3)
    y2 = _rope_native(z2, cos_sin)
    return (
        x2.reshape(batch_size, seq_len, hidden_dim),
        y2.reshape(batch_size, seq_len, z2.shape[-1]),
    )


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


def _transformer_gemm_flops(args: argparse.Namespace) -> int:
    m = args.batch_size * args.seq_len
    hidden_dim = args.hidden_dim
    intermediate_dim = args.intermediate_dim
    qkv_dim = args.qkv_dim or hidden_dim * 3
    return 2 * m * (
        hidden_dim * hidden_dim
        + hidden_dim * (intermediate_dim * 2)
        + intermediate_dim * hidden_dim
        + hidden_dim * qkv_dim
    )


def _gemm_gflops(flops: int, elapsed_ms: float) -> float:
    return flops / (elapsed_ms * 1_000_000.0)


def _warm_result(flops: int, median_ms: float) -> str:
    return f"warm_median={median_ms:.3f} ms gemm_gflops={_gemm_gflops(flops, median_ms):.2f}"


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


def _make_inputs(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    dtype = torch.float32 if getattr(args, "dtype", "float32") == "float32" else torch.bfloat16
    qkv_dim = args.qkv_dim or args.hidden_dim * 3

    x0 = torch.randn((args.batch_size, args.seq_len, args.hidden_dim), dtype=dtype)
    y0 = torch.randn((args.batch_size, args.seq_len, args.hidden_dim), dtype=dtype)
    w0 = torch.randn((args.hidden_dim, args.hidden_dim), dtype=dtype) / args.hidden_dim**0.5
    w1 = torch.randn((args.hidden_dim, args.intermediate_dim * 2), dtype=dtype) / args.hidden_dim**0.5
    w2 = torch.randn((args.intermediate_dim, args.hidden_dim), dtype=dtype) / args.intermediate_dim**0.5
    w3 = torch.randn((args.hidden_dim, qkv_dim), dtype=dtype) / args.hidden_dim**0.5
    wn0 = torch.randn((args.hidden_dim,), dtype=dtype)
    wn1 = torch.randn((args.hidden_dim,), dtype=dtype)
    cos_sin = torch.empty((args.batch_size * args.seq_len, qkv_dim), dtype=dtype)
    cos_sin[:, 0::2] = 1.0
    cos_sin[:, 1::2] = 0.0
    cos = torch.ones((args.seq_len, args.head_dim // 2), dtype=dtype)
    sin = torch.zeros((args.seq_len, args.head_dim // 2), dtype=dtype)
    return x0, y0, w0, w1, w2, w3, wn0, wn1, cos_sin, cos, sin


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a CPU CODA transformer layer.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=2560)
    parser.add_argument("--intermediate-dim", type=int, default=4096)
    parser.add_argument("--qkv-dim", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=20)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-compiled", action="store_true")
    parser.add_argument("--dtype", type=str, choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    inputs = _make_inputs(args)
    x0, y0, w0, w1, w2, w3, wn0, wn1, cos_sin, cos, sin = inputs
    gemm_flops = _transformer_gemm_flops(args)
    print(f"workload: gemm_flops={gemm_flops / 1e9:.6f} GFLOP")

    def cpu_layer_fn(backend: str):
        return lambda: ops.layer(
            x0=x0,
            y0=y0,
            w0=w0,
            w1=w1,
            w2=w2,
            w3=w3,
            wn0=wn0,
            wn1=wn1,
            cos_sin=cos_sin,
            cos=cos,
            sin=sin,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            eps=1e-6,
            transpose=False,
            backend=backend,
            use_compile=False,
        )

    torch_fn = lambda: _torch_native_layer(
        x0=x0,
        y0=y0,
        w0=w0,
        w1=w1,
        w2=w2,
        w3=w3,
        wn0=wn0,
        wn1=wn1,
        cos_sin=cos_sin,
        eps=1e-6,
    )
    torch_out = torch_fn()
    cold_ms = _time_once(torch_fn)
    median_ms = _bench(torch_fn, warmup=args.warmup, repeats=args.repeats)
    print(f"torch/native: cold={cold_ms:.3f} ms {_warm_result(gemm_flops, median_ms)}")

    if not args.skip_compiled:
        try:
            compiled_layer = torch.compile(_torch_native_layer, fullgraph=True, dynamic=False)
            torch_compiled_fn = lambda: compiled_layer(
                x0=x0,
                y0=y0,
                w0=w0,
                w1=w1,
                w2=w2,
                w3=w3,
                wn0=wn0,
                wn1=wn1,
                cos_sin=cos_sin,
                eps=1e-6,
            )
            compiled_out = torch_compiled_fn()
            max_diff = max(
                (actual.float() - expected.float()).abs().max().item()
                for actual, expected in zip(compiled_out, torch_out)
            )
            cold_ms = _time_once(torch_compiled_fn)
            median_ms = _bench(torch_compiled_fn, warmup=args.warmup, repeats=args.repeats)
            print(
                f"torch/compiled: cold={cold_ms:.3f} ms "
                f"{_warm_result(gemm_flops, median_ms)} max_diff={max_diff:.6f}"
            )
        except Exception as exc:
            print(f"torch/compiled: skipped ({type(exc).__name__}: {exc})")

    for provider in ("aten", "aten-vec", "onednn-x64-brgemm", "libxsmm"):
        try:
            select_provider(provider)
        except RuntimeError as exc:
            print(f"cpu/{provider}: skipped ({exc})")
            continue

        with _provider_env(provider):
            cpu_fn = cpu_layer_fn("cpu")
            cpu_out = cpu_fn()
            max_diff = max(
                (actual.float() - expected.float()).abs().max().item()
                for actual, expected in zip(cpu_out, torch_out)
            )
            cold_ms = _time_once(cpu_fn)
            median_ms = _bench(cpu_fn, warmup=args.warmup, repeats=args.repeats)
            print(
                f"cpu/{provider}: cold={cold_ms:.3f} ms "
                f"{_warm_result(gemm_flops, median_ms)} max_diff={max_diff:.6f}"
            )


if __name__ == "__main__":
    main()
