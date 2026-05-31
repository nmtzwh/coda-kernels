from __future__ import annotations

import importlib.util
import os
import platform
from dataclasses import dataclass

import torch

from kernels.cpu.ir import EpilogueOp, GemmEpilogueProgram


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _require_matrix(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor, got shape={tuple(tensor.shape)}")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU for backend='cpu', got device={tensor.device}")


def _to_acc(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=torch.float32)


def _row_view(tensor: torch.Tensor) -> torch.Tensor:
    return _to_acc(tensor).reshape(-1, 1)


def _col_view(tensor: torch.Tensor) -> torch.Tensor:
    return _to_acc(tensor).reshape(1, -1)


def _rope(x: torch.Tensor, cos_sin: torch.Tensor, backward: bool = False) -> torch.Tensor:
    if x.shape != cos_sin.shape:
        raise ValueError(
            f"cos_sin must match GEMM output shape, got {tuple(cos_sin.shape)} vs {tuple(x.shape)}"
        )
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE expects an even last dimension, got {x.shape[-1]}")

    sign = -1.0 if backward else 1.0
    x2 = _to_acc(x).reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    cs2 = _to_acc(cos_sin).reshape(*cos_sin.shape[:-1], cos_sin.shape[-1] // 2, 2)
    x0 = x2[..., 0]
    x1 = x2[..., 1]
    cos = cs2[..., 0]
    sin = cs2[..., 1] * sign
    y0 = x0 * cos + x1 * sin
    y1 = x0 * (-sin) + x1 * cos
    return torch.stack((y0, y1), dim=-1).reshape_as(x)


def _swiglu(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"SwiGLU expects an even last dimension, got {x.shape[-1]}")
    x2 = _to_acc(x).reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    gate = x2[..., 0]
    up = x2[..., 1]
    return torch.nn.functional.silu(gate) * up


@dataclass(frozen=True)
class CpuBackendFeatures:
    machine: str
    processor: str
    has_libxsmm_python: bool
    has_onednn_python: bool


def detect_features() -> CpuBackendFeatures:
    return CpuBackendFeatures(
        machine=platform.machine().lower(),
        processor=platform.processor().lower(),
        has_libxsmm_python=importlib.util.find_spec("libxsmm") is not None,
        has_onednn_python=any(
            importlib.util.find_spec(name) is not None
            for name in ("onednn", "dnnl")
        ),
    )


class CpuGemmProvider:
    name = "base"

    @classmethod
    def is_available(cls, features: CpuBackendFeatures) -> bool:
        return False

    def matmul_accumulate(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        _require_matrix("A", A)
        _require_matrix("B", B)
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"incompatible GEMM shapes: {tuple(A.shape)} x {tuple(B.shape)}")
        return _to_acc(A).mm(_to_acc(B))

    def execute(
        self,
        program: GemmEpilogueProgram,
        A: torch.Tensor,
        B: torch.Tensor,
        **tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        acc = self.matmul_accumulate(A, B)
        side_outputs: dict[str, torch.Tensor] = {}

        for node in program.nodes:
            if node.op == EpilogueOp.RESIDUAL_ADD:
                aux = _to_acc(tensors[_require_name(node.tensor)])
                if aux.shape != acc.shape:
                    raise ValueError(f"residual shape mismatch: {tuple(aux.shape)} vs {tuple(acc.shape)}")
                acc = acc + aux

            elif node.op == EpilogueOp.ROW_SCALE:
                acc = acc * _row_view(tensors[_require_name(node.tensor)])

            elif node.op == EpilogueOp.ROW_VECTOR_SCALE_SIDE_OUTPUT:
                side_outputs[_require_name(node.output)] = acc * _col_view(tensors[_require_name(node.tensor)])

            elif node.op == EpilogueOp.BLOCK_MEAN_SQUARE_REDUCTION:
                if node.block_size is None:
                    raise ValueError("block mean-square reduction requires block_size")
                if acc.shape[1] % node.block_size != 0:
                    raise ValueError(
                        f"N={acc.shape[1]} must be divisible by block_size={node.block_size}"
                    )
                side_outputs[_require_name(node.output)] = (
                    acc.square()
                    .reshape(acc.shape[0], _ceil_div(acc.shape[1], node.block_size), node.block_size)
                    .mean(dim=-1)
                )

            elif node.op == EpilogueOp.SWIGLU:
                side_outputs[_require_name(node.output)] = _swiglu(acc)

            elif node.op == EpilogueOp.ROPE:
                side_outputs[_require_name(node.output)] = _rope(
                    acc,
                    tensors[_require_name(node.tensor)],
                    backward=node.backward,
                )

            elif node.op == EpilogueOp.TARGET_LOGIT_SELECT:
                targets = tensors[_require_name(node.tensor)].to(dtype=torch.long)
                if targets.ndim != 1 or targets.shape[0] != acc.shape[0]:
                    raise ValueError(
                        f"targets must have shape ({acc.shape[0]},), got {tuple(targets.shape)}"
                    )
                rows = torch.arange(acc.shape[0], device=acc.device)
                side_outputs[_require_name(node.output)] = acc[rows, targets]

            elif node.op == EpilogueOp.BLOCK_LOGSUMEXP:
                if node.block_size is None:
                    raise ValueError("block logsumexp requires block_size")
                if acc.shape[1] % node.block_size != 0:
                    raise ValueError(
                        f"N={acc.shape[1]} must be divisible by block_size={node.block_size}"
                    )
                side_outputs[_require_name(node.output)] = torch.logsumexp(
                    acc.reshape(acc.shape[0], _ceil_div(acc.shape[1], node.block_size), node.block_size),
                    dim=-1,
                )

            else:
                raise NotImplementedError(f"unsupported CPU epilogue op: {node.op}")

        output_dtype = program.output_dtype or A.dtype
        return acc.to(dtype=output_dtype), side_outputs


def _require_name(value: str | None) -> str:
    if value is None:
        raise ValueError("epilogue node is missing a tensor/output name")
    return value


class AtenFallbackProvider(CpuGemmProvider):
    name = "aten"

    @classmethod
    def is_available(cls, features: CpuBackendFeatures) -> bool:
        return True


class OneDnnX64BrgemmProvider(CpuGemmProvider):
    name = "onednn-x64-brgemm"

    @classmethod
    def is_available(cls, features: CpuBackendFeatures) -> bool:
        # Placeholder until the native oneDNN ukernel bridge is added.
        return False


class LibxsmmProvider(CpuGemmProvider):
    name = "libxsmm"

    @classmethod
    def is_available(cls, features: CpuBackendFeatures) -> bool:
        # Placeholder until the native LIBXSMM dispatch bridge is added.
        return False


_PROVIDER_CLASSES: tuple[type[CpuGemmProvider], ...] = (
    OneDnnX64BrgemmProvider,
    LibxsmmProvider,
    AtenFallbackProvider,
)


def select_provider(preferred: str | None = None) -> CpuGemmProvider:
    features = detect_features()
    requested = preferred or os.environ.get("CODA_CPU_PROVIDER")

    if requested:
        requested = requested.lower()
        for provider_cls in _PROVIDER_CLASSES:
            if requested in {provider_cls.name, provider_cls.name.split("-")[0]}:
                if not provider_cls.is_available(features):
                    raise RuntimeError(
                        f"CPU provider '{provider_cls.name}' is not available in this build; "
                        "use CODA_CPU_PROVIDER=aten or build the native provider."
                    )
                return provider_cls()
        raise ValueError(f"unknown CPU provider: {requested}")

    for provider_cls in _PROVIDER_CLASSES:
        if provider_cls.is_available(features):
            return provider_cls()

    return AtenFallbackProvider()


def current_provider_name() -> str:
    return select_provider().name
