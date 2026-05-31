import math

import pytest

torch = pytest.importorskip("torch")

from kernels.cpu import gpt as cpu_gpt
from kernels.refs import gpt2 as ref_gpt
from models import ops


def _randn(shape, dtype):
    return torch.randn(shape, dtype=torch.float32).to(dtype=dtype)


def _rope_identity(m: int, n: int, dtype: torch.dtype) -> torch.Tensor:
    cos = torch.ones((m, n // 2), dtype=torch.float32)
    sin = torch.zeros((m, n // 2), dtype=torch.float32)
    return torch.stack((cos, sin), dim=-1).reshape(m, n).to(dtype=dtype)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    else:
        torch.testing.assert_close(actual.float(), expected.float(), rtol=8e-3, atol=8e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_forward_kernels_match_torch_reference(dtype: torch.dtype) -> None:
    M, K, N = 8, 16, 32
    block_size = 8
    A = _randn((M, K), dtype)
    B = _randn((K, N), dtype) / math.sqrt(K)
    C = _randn((M, N), dtype)
    W = _randn((N,), dtype)
    R = torch.rsqrt((A.float() ** 2).mean(dim=-1) + 1e-6)
    cos_sin = _rope_identity(M, N, dtype)

    D, S, O = cpu_gpt.gemm_residual_partial_rmsnorm(
        A=A,
        B=B,
        C=C,
        W=W,
        block_size=block_size,
    )
    D_ref, S_ref, O_ref = ref_gpt.gemm_residual_partial_rmsnorm(
        A=A,
        B=B,
        C=C,
        W=W,
        block_size=block_size,
    )
    _assert_close(D, D_ref, dtype)
    torch.testing.assert_close(S, S_ref.float(), rtol=8e-3, atol=8e-3)
    _assert_close(O, O_ref, dtype)

    _assert_close(cpu_gpt.gemm_rmsnorm(A, B, R), ref_gpt.gemm_rmsnorm(A, B, R), dtype)

    D, O = cpu_gpt.gemm_swiglu(A, B)
    D_ref, O_ref = ref_gpt.gemm_swiglu(A, B)
    _assert_close(D, D_ref, dtype)
    _assert_close(O, O_ref, dtype)

    D, O = cpu_gpt.gemm_rmsnorm_swiglu(A, B, R)
    D_ref, O_ref = ref_gpt.gemm_rmsnorm_swiglu(A, B, R)
    _assert_close(D, D_ref, dtype)
    _assert_close(O, O_ref, dtype)

    D, O = cpu_gpt.gemm_rope(A, B, cos_sin)
    D_ref, O_ref = ref_gpt.gemm_rope(A, B, cos_sin)
    _assert_close(D, D_ref, dtype)
    _assert_close(O, O_ref, dtype)

    D, O = cpu_gpt.gemm_rmsnorm_rope(A, B, R, cos_sin)
    D_ref, O_ref = ref_gpt.gemm_rmsnorm_rope(A, B, R, cos_sin)
    _assert_close(D, D_ref, dtype)
    _assert_close(O, O_ref, dtype)


def test_ops_layer_cpu_forward_matches_torch_backend() -> None:
    dtype = torch.float32
    batch_size, seq_len = 2, 4
    hidden_dim, mlp_dim = 16, 32
    qkv_dim = 48
    num_heads, head_dim = 2, 8

    x0 = _randn((batch_size, seq_len, hidden_dim), dtype)
    y0 = _randn((batch_size, seq_len, hidden_dim), dtype)
    w0 = _randn((hidden_dim, hidden_dim), dtype) / math.sqrt(hidden_dim)
    w1 = _randn((hidden_dim, mlp_dim * 2), dtype) / math.sqrt(hidden_dim)
    w2 = _randn((mlp_dim, hidden_dim), dtype) / math.sqrt(mlp_dim)
    w3 = _randn((hidden_dim, qkv_dim), dtype) / math.sqrt(hidden_dim)
    wn0 = _randn((hidden_dim,), dtype)
    wn1 = _randn((hidden_dim,), dtype)
    cos_sin = _rope_identity(batch_size * seq_len, qkv_dim, dtype)
    cos = torch.ones((seq_len, head_dim // 2), dtype=dtype)
    sin = torch.zeros((seq_len, head_dim // 2), dtype=dtype)

    cpu_out = ops.layer(
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
        num_heads=num_heads,
        head_dim=head_dim,
        eps=1e-6,
        transpose=False,
        backend="cpu",
        use_compile=False,
    )
    torch_out = ops.layer(
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
        num_heads=num_heads,
        head_dim=head_dim,
        eps=1e-6,
        transpose=False,
        backend="torch",
        use_compile=False,
    )

    for actual, expected in zip(cpu_out, torch_out):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
