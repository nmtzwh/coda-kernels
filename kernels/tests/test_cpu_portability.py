import argparse
from pathlib import Path

import pytest

from kernels.benchmarks.cpu_transformer import _gemm_gflops, _transformer_gemm_flops
from kernels.cpu import native, providers


class _NativeProviders:
    @staticmethod
    def has_onednn() -> bool:
        return True

    @staticmethod
    def has_libxsmm() -> bool:
        return True

    @staticmethod
    def has_aten_vec() -> bool:
        return True


def _features(
    machine: str,
    *,
    has_avx2: bool = False,
    has_avx512: bool = False,
    has_neon: bool = False,
    has_sve: bool = False,
) -> providers.CpuBackendFeatures:
    return providers.CpuBackendFeatures(
        machine=machine,
        processor="",
        torch_cpu_capability="",
        flags=frozenset(),
        has_avx2=has_avx2,
        has_avx512=has_avx512,
        has_amx=False,
        has_neon=has_neon,
        has_sve=has_sve,
        has_libxsmm_python=False,
        has_onednn_python=False,
    )


def test_transformer_default_gemm_flops() -> None:
    args = argparse.Namespace(
        batch_size=1,
        seq_len=128,
        hidden_dim=2560,
        intermediate_dim=4096,
        qkv_dim=None,
    )
    flops = _transformer_gemm_flops(args)
    assert flops == 14_763_950_080
    assert _gemm_gflops(flops, 100.0) == pytest.approx(147.6395008)


def test_cpuinfo_feature_parser_accepts_x86_and_aarch64_keys() -> None:
    assert providers._parse_cpu_flags("flags : fpu avx2 avx512f") == frozenset(
        {"fpu", "avx2", "avx512f"}
    )
    assert providers._parse_cpu_flags("Features : fp asimd sve") == frozenset(
        {"fp", "asimd", "sve"}
    )


def test_native_provider_architecture_gates_include_supported_aarch64(monkeypatch) -> None:
    monkeypatch.setattr(providers, "load_native_extension", lambda: _NativeProviders())

    assert providers.OneDnnX64BrgemmProvider.is_available(
        _features("x86_64", has_avx2=True)
    )
    assert providers.OneDnnX64BrgemmProvider.is_available(
        _features("aarch64", has_sve=True)
    )
    assert not providers.OneDnnX64BrgemmProvider.is_available(
        _features("aarch64", has_neon=True)
    )

    assert providers.LibxsmmProvider.is_available(_features("x86_64", has_avx2=True))
    assert providers.LibxsmmProvider.is_available(_features("aarch64", has_neon=True))
    assert providers.LibxsmmProvider.is_available(_features("aarch64", has_sve=True))
    assert not providers.LibxsmmProvider.is_available(_features("aarch64"))


def test_aten_vec_auto_isa_prefers_widest_supported_specialization(monkeypatch) -> None:
    monkeypatch.setattr(native, "_supports_avx512", lambda: True)
    monkeypatch.setattr(native, "_supports_avx2", lambda: True)
    monkeypatch.setattr(native, "_supports_sve256", lambda: False)
    assert native._select_aten_vec_isa("auto") == "avx512"

    monkeypatch.setattr(native, "_supports_avx512", lambda: False)
    assert native._select_aten_vec_isa("auto") == "avx2"

    monkeypatch.setattr(native, "_supports_avx2", lambda: False)
    monkeypatch.setattr(native, "_supports_sve256", lambda: True)
    assert native._select_aten_vec_isa("auto") == "sve256"

    monkeypatch.setattr(native, "_supports_sve256", lambda: False)
    assert native._select_aten_vec_isa("auto") == "generic"


def test_aten_vec_kernel_uses_generic_aten_vector_api() -> None:
    source = (
        Path(__file__).parents[1] / "cpu" / "native" / "coda_aten_vec.cpp"
    ).read_text(encoding="utf-8")
    assert "<ATen/cpu/vec/vec.h>" in source
    assert "at::vec::Vectorized<float>" in source
    for architecture_specific_token in (
        "<immintrin.h>",
        "<arm_neon.h>",
        "_mm256",
        "_mm512",
        "svfloat",
    ):
        assert architecture_specific_token not in source
