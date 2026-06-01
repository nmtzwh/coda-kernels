import os
import math

import pytest

torch = pytest.importorskip("torch")

from kernels.cpu import gpt as cpu_gpt
from kernels.cpu.ir import gelu_tanh, row_bias, scalar_scale, store_accumulator
from kernels.cpu.native import load_native_extension
from kernels.cpu.providers import LibxsmmProvider, OneDnnX64BrgemmProvider, select_provider
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
    hidden_dim, mlp_dim = 128, 256
    qkv_dim = 384
    num_heads, head_dim = 4, 32

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


def test_transformer_post_ops_serialize_for_native_provider() -> None:
    nodes = [
        row_bias("bias"),
        scalar_scale(0.5),
        gelu_tanh(),
        store_accumulator("D"),
    ]
    native_nodes = [node.to_native() for node in nodes]
    assert [node["op"] for node in native_nodes] == [
        "row_bias",
        "scalar_scale",
        "gelu_tanh",
        "store_accumulator",
    ]
    assert native_nodes[1]["value"] == 0.5
    assert native_nodes[3]["output"] == "D"


def test_onednn_provider_requires_native_onednn_build(monkeypatch) -> None:
    native = load_native_extension()
    if native is not None and native.has_onednn():
        pytest.skip("oneDNN native provider is available in this environment")

    monkeypatch.setenv("CODA_CPU_PROVIDER", OneDnnX64BrgemmProvider.name)
    with pytest.raises(RuntimeError):
        select_provider()


def test_libxsmm_provider_requires_native_libxsmm_build(monkeypatch) -> None:
    native = load_native_extension()
    if native is not None and hasattr(native, "has_libxsmm") and native.has_libxsmm():
        pytest.skip("LIBXSMM native provider is available in this environment")

    monkeypatch.setenv("CODA_CPU_PROVIDER", LibxsmmProvider.name)
    with pytest.raises(RuntimeError):
        select_provider()


def test_optional_native_post_op_chain_smoke() -> None:
    if os.environ.get("CODA_CPU_JIT_BUILD") != "1":
        pytest.skip("native extension JIT build not requested")

    native = load_native_extension()
    if native is None:
        pytest.skip("native extension is not available")

    A = torch.randn((4, 8), dtype=torch.float32)
    B = torch.randn((8, 16), dtype=torch.float32)
    R = torch.rsqrt((A ** 2).mean(dim=-1) + 1e-6)
    program = cpu_gpt.GemmEpilogueProgram(
        name="native_smoke",
        nodes=(cpu_gpt.row_scale("R"), cpu_gpt.swiglu("O")),
        output_dtype=A.dtype,
    )
    D, outputs = native.execute_brgemm_postops(
        program.name,
        program.to_native(),
        A,
        B,
        {"R": R},
    )
    D_ref = A.float().mm(B.float()) * R.float().reshape(-1, 1)
    D2_ref = D_ref.reshape(4, 8, 2)
    O_ref = torch.nn.functional.silu(D2_ref[..., 0]) * D2_ref[..., 1]
    torch.testing.assert_close(D, D_ref)
    torch.testing.assert_close(outputs["O"], O_ref)


def test_optional_native_libxsmm_post_op_chain_smoke(monkeypatch) -> None:
    if os.environ.get("CODA_CPU_JIT_BUILD") != "1":
        pytest.skip("native extension JIT build not requested")
    if os.environ.get("CODA_CPU_WITH_LIBXSMM") != "1":
        pytest.skip("LIBXSMM JIT build not requested")

    native = load_native_extension()
    if native is None or not hasattr(native, "has_libxsmm") or not native.has_libxsmm():
        pytest.skip("LIBXSMM headers/libs are not available")

    monkeypatch.setenv("CODA_CPU_PROVIDER", LibxsmmProvider.name)
    monkeypatch.setenv("CODA_LIBXSMM_DENSE_ATEN", "0")
    monkeypatch.setenv("CODA_LIBXSMM_DENSE_MIN_FLOPS", "1")
    provider = select_provider()
    assert provider.name == LibxsmmProvider.name

    A = torch.randn((4, 8), dtype=torch.float32)
    B = torch.randn((8, 16), dtype=torch.float32)
    R = torch.rsqrt((A ** 2).mean(dim=-1) + 1e-6)
    D, O = cpu_gpt.gemm_rmsnorm_swiglu(A, B, R)
    D_ref = A.float().mm(B.float()) * R.float().reshape(-1, 1)
    D2_ref = D_ref.reshape(4, 8, 2)
    O_ref = torch.nn.functional.silu(D2_ref[..., 0]) * D2_ref[..., 1]
    torch.testing.assert_close(D, D_ref)
    torch.testing.assert_close(O, O_ref)
