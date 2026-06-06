from __future__ import annotations

import importlib.util
import os
import platform
from dataclasses import dataclass
from functools import lru_cache

import torch

from kernels.cpu.ir import EpilogueOp, GemmEpilogueProgram
from kernels.cpu.native import load_native_extension


_X86_64_MACHINES = frozenset({"x86_64", "amd64", "x64"})
_AARCH64_MACHINES = frozenset({"aarch64", "arm64"})


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


def _scalar_value(value: float | None) -> float:
    if value is None:
        raise ValueError("epilogue node is missing a scalar value")
    return float(value)


def _rope(x: torch.Tensor, cos_sin: torch.Tensor, backward: bool = False) -> torch.Tensor:
    if x.shape != cos_sin.shape:
        raise ValueError(
            f"cos_sin must match GEMM output shape, got {tuple(cos_sin.shape)} vs {tuple(x.shape)}"
        )
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE expects an even last dimension, got {x.shape[-1]}")

    sign = -1.0 if backward else 1.0
    x_acc = _to_acc(x)
    x2 = x_acc.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    cs2 = _to_acc(cos_sin).reshape(*cos_sin.shape[:-1], cos_sin.shape[-1] // 2, 2)
    x0 = x2[..., 0]
    x1 = x2[..., 1]
    cos = cs2[..., 0]
    sin = cs2[..., 1] * sign
    out = x_acc.new_empty(x_acc.shape)
    out2 = out.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    y0 = out2[..., 0]
    y1 = out2[..., 1]
    torch.mul(x0, cos, out=y0)
    y0.addcmul_(x1, sin)
    torch.mul(x1, cos, out=y1)
    y1.addcmul_(x0, sin, value=-1.0)
    return out


def _swiglu(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"SwiGLU expects an even last dimension, got {x.shape[-1]}")
    x2 = _to_acc(x).reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    gate = x2[..., 0]
    up = x2[..., 1]
    out = gate.new_empty(gate.shape)
    torch.sigmoid(gate, out=out)
    out.mul_(gate)
    out.mul_(up)
    return out


@dataclass(frozen=True)
class CpuBackendFeatures:
    machine: str
    processor: str
    torch_cpu_capability: str
    flags: frozenset[str]
    has_avx2: bool
    has_avx512: bool
    has_amx: bool
    has_neon: bool
    has_sve: bool
    has_libxsmm_python: bool
    has_onednn_python: bool


def _parse_cpu_flags(cpuinfo: str) -> frozenset[str]:
    for line in cpuinfo.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() in {"flags", "features"}:
            return frozenset(value.lower().split())
    return frozenset()


def _cpu_flags() -> frozenset[str]:
    if os.name == "posix":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
                return _parse_cpu_flags(cpuinfo.read())
        except OSError:
            pass
    return frozenset()


def _torch_cpu_capability() -> str:
    try:
        get_capability = getattr(torch.backends.cpu, "get_cpu_capability", None)
        if get_capability is None:
            return ""
        return str(get_capability()).lower()
    except Exception:
        return ""


@lru_cache(maxsize=1)
def detect_features() -> CpuBackendFeatures:
    flags = _cpu_flags()
    torch_capability = _torch_cpu_capability()
    machine = platform.machine().lower()
    return CpuBackendFeatures(
        machine=machine,
        processor=platform.processor().lower(),
        torch_cpu_capability=torch_capability,
        flags=flags,
        has_avx2="avx2" in flags or "avx2" in torch_capability or "avx512" in torch_capability,
        has_avx512=any(flag.startswith("avx512") for flag in flags) or "avx512" in torch_capability,
        has_amx=any(flag.startswith("amx") for flag in flags),
        has_neon=machine in _AARCH64_MACHINES and (
            "asimd" in flags or "neon" in flags or platform.system() == "Darwin"
        ),
        has_sve="sve" in flags or "sve" in torch_capability,
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
        if (
            len(program.nodes) == 3
            and program.nodes[0].op == EpilogueOp.RESIDUAL_ADD
            and program.nodes[1].op == EpilogueOp.BLOCK_MEAN_SQUARE_REDUCTION
            and program.nodes[2].op == EpilogueOp.ROW_VECTOR_SCALE_SIDE_OUTPUT
        ):
            c = _to_acc(tensors[_require_name(program.nodes[0].tensor)])
            w = _col_view(tensors[_require_name(program.nodes[2].tensor)])
            block_size = program.nodes[1].block_size
            if block_size is None:
                raise ValueError("block mean-square reduction requires block_size")
            acc = torch.addmm(c, _to_acc(A), _to_acc(B))
            if acc.shape[1] % block_size != 0:
                raise ValueError(f"N={acc.shape[1]} must be divisible by block_size={block_size}")
            side_outputs = {
                _require_name(program.nodes[1].output): (
                    acc.square()
                    .reshape(acc.shape[0], _ceil_div(acc.shape[1], block_size), block_size)
                    .mean(dim=-1)
                ),
                _require_name(program.nodes[2].output): acc * w,
            }
            output_dtype = program.output_dtype or A.dtype
            return acc.to(dtype=output_dtype), side_outputs

        if len(program.nodes) == 1 and program.nodes[0].op == EpilogueOp.ROW_SCALE:
            acc = self.matmul_accumulate(A, B)
            acc.mul_(_row_view(tensors[_require_name(program.nodes[0].tensor)]))
            output_dtype = program.output_dtype or A.dtype
            return acc.to(dtype=output_dtype), {}

        if (
            len(program.nodes) == 2
            and program.nodes[0].op == EpilogueOp.ROW_SCALE
            and program.nodes[1].op == EpilogueOp.SWIGLU
        ):
            acc = self.matmul_accumulate(A, B)
            acc.mul_(_row_view(tensors[_require_name(program.nodes[0].tensor)]))
            side_outputs = {_require_name(program.nodes[1].output): _swiglu(acc)}
            output_dtype = program.output_dtype or A.dtype
            return acc.to(dtype=output_dtype), side_outputs

        if (
            len(program.nodes) == 2
            and program.nodes[0].op == EpilogueOp.ROW_SCALE
            and program.nodes[1].op == EpilogueOp.ROPE
        ):
            acc = self.matmul_accumulate(A, B)
            acc.mul_(_row_view(tensors[_require_name(program.nodes[0].tensor)]))
            side_outputs = {
                _require_name(program.nodes[1].output): _rope(
                    acc,
                    tensors[_require_name(program.nodes[1].tensor)],
                    backward=program.nodes[1].backward,
                )
            }
            output_dtype = program.output_dtype or A.dtype
            return acc.to(dtype=output_dtype), side_outputs

        acc = self.matmul_accumulate(A, B)
        side_outputs: dict[str, torch.Tensor] = {}

        for node in program.nodes:
            if node.op == EpilogueOp.RESIDUAL_ADD:
                aux = _to_acc(tensors[_require_name(node.tensor)])
                if aux.shape != acc.shape:
                    raise ValueError(f"residual shape mismatch: {tuple(aux.shape)} vs {tuple(acc.shape)}")
                acc = acc + aux

            elif node.op == EpilogueOp.ROW_BIAS:
                acc = acc + _row_view(tensors[_require_name(node.tensor)])

            elif node.op == EpilogueOp.COL_BIAS:
                acc = acc + _col_view(tensors[_require_name(node.tensor)])

            elif node.op == EpilogueOp.SCALAR_SCALE:
                acc = acc * _scalar_value(node.value)

            elif node.op == EpilogueOp.ROW_SCALE:
                acc = acc * _row_view(tensors[_require_name(node.tensor)])

            elif node.op == EpilogueOp.COL_SCALE:
                acc = acc * _col_view(tensors[_require_name(node.tensor)])

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

            elif node.op == EpilogueOp.ROW_SUM:
                side_outputs[_require_name(node.output)] = acc.sum(dim=-1)

            elif node.op == EpilogueOp.ROW_SUM_SQUARES:
                side_outputs[_require_name(node.output)] = acc.square().sum(dim=-1)

            elif node.op == EpilogueOp.BLOCK_MAX:
                if node.block_size is None:
                    raise ValueError("block max requires block_size")
                if acc.shape[1] % node.block_size != 0:
                    raise ValueError(
                        f"N={acc.shape[1]} must be divisible by block_size={node.block_size}"
                    )
                side_outputs[_require_name(node.output)] = (
                    acc.reshape(acc.shape[0], _ceil_div(acc.shape[1], node.block_size), node.block_size)
                    .max(dim=-1)
                    .values
                )

            elif node.op == EpilogueOp.SILU:
                acc = torch.nn.functional.silu(acc)

            elif node.op == EpilogueOp.GELU_TANH:
                acc = torch.nn.functional.gelu(acc, approximate="tanh")

            elif node.op == EpilogueOp.RELU:
                acc = torch.relu(acc)

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

            elif node.op == EpilogueOp.STORE_ACCUMULATOR:
                side_outputs[_require_name(node.output)] = acc

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


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off", ""}


def _is_contiguous_f32(tensor: object) -> bool:
    if not isinstance(tensor, torch.Tensor):
        return False
    return (
        tensor.device.type == "cpu"
        and tensor.dtype == torch.float32
        and tensor.is_contiguous()
    )


def _is_contiguous_i64(tensor: object) -> bool:
    if not isinstance(tensor, torch.Tensor):
        return False
    return (
        tensor.device.type == "cpu"
        and tensor.dtype == torch.long
        and tensor.is_contiguous()
    )


def _is_contiguous_f32_or_bf16(tensor: object) -> bool:
    if not isinstance(tensor, torch.Tensor):
        return False
    return (
        tensor.device.type == "cpu"
        and tensor.dtype in {torch.float32, torch.bfloat16}
        and tensor.is_contiguous()
    )


def _is_contiguous_of_dtype(tensor: object, dtype: torch.dtype) -> bool:
    if not isinstance(tensor, torch.Tensor):
        return False
    return (
        tensor.device.type == "cpu"
        and tensor.dtype == dtype
        and tensor.is_contiguous()
    )


class AtenVecProvider(CpuGemmProvider):
    name = "aten-vec"

    def __init__(self):
        super().__init__()

    @classmethod
    def is_available(cls, features: CpuBackendFeatures) -> bool:
        native = load_native_extension()
        return (
            native is not None
            and hasattr(native, "has_aten_vec")
            and bool(native.has_aten_vec())
        )

    def _is_supported(
        self,
        program: GemmEpilogueProgram,
        A: torch.Tensor,
        B: torch.Tensor,
        tensors: dict[str, torch.Tensor],
    ) -> bool:
        if not _is_contiguous_f32_or_bf16(A) or not _is_contiguous_f32_or_bf16(B):
            return False
        if A.dtype != B.dtype:
            return False

        def check(name: str | None) -> bool:
            return _is_contiguous_f32_or_bf16(tensors.get(_require_name(name)))

        nodes = program.nodes
        if (
            len(nodes) == 3
            and nodes[0].op == EpilogueOp.RESIDUAL_ADD
            and nodes[1].op == EpilogueOp.BLOCK_MEAN_SQUARE_REDUCTION
            and nodes[2].op == EpilogueOp.ROW_VECTOR_SCALE_SIDE_OUTPUT
        ):
            return (
                check(nodes[0].tensor)
                and check(nodes[2].tensor)
            )

        if len(nodes) == 1 and nodes[0].op in {
            EpilogueOp.ROW_SCALE,
            EpilogueOp.SWIGLU,
            EpilogueOp.ROPE,
        }:
            if nodes[0].op == EpilogueOp.ROW_SCALE:
                return check(nodes[0].tensor)
            if nodes[0].op == EpilogueOp.ROPE:
                return check(nodes[0].tensor)
            return True

        if len(nodes) == 2 and nodes[0].op == EpilogueOp.ROW_SCALE and nodes[1].op in {
            EpilogueOp.SWIGLU,
            EpilogueOp.ROPE,
        }:
            if not check(nodes[0].tensor):
                return False
            if nodes[1].op == EpilogueOp.ROPE:
                return check(nodes[1].tensor)
            return True

        if (
            len(nodes) == 2
            and nodes[0].op == EpilogueOp.TARGET_LOGIT_SELECT
            and nodes[1].op == EpilogueOp.BLOCK_LOGSUMEXP
        ):
            return _is_contiguous_i64(tensors.get(_require_name(nodes[0].tensor)))

        if (
            len(nodes) == 3
            and nodes[0].op == EpilogueOp.ROW_SCALE
            and nodes[1].op == EpilogueOp.TARGET_LOGIT_SELECT
            and nodes[2].op == EpilogueOp.BLOCK_LOGSUMEXP
        ):
            return (
                check(nodes[0].tensor)
                and _is_contiguous_i64(tensors.get(_require_name(nodes[1].tensor)))
            )

        return False

    def execute(
        self,
        program: GemmEpilogueProgram,
        A: torch.Tensor,
        B: torch.Tensor,
        **tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        native = load_native_extension()
        supported = (
            native is not None
            and hasattr(native, "execute_aten_vec_postops")
            and self._is_supported(
                program,
                A,
                B,
                tensors,
            )
        )
        if not supported:
            if _env_enabled("CODA_ATEN_VEC_STRICT"):
                raise RuntimeError(
                    "CPU provider 'aten-vec' only supports forward contiguous float32/bfloat16 inputs "
                    "for the current transformer epilogue programs"
                )
            return super().execute(program, A, B, **tensors)

        out, side_outputs = native.execute_aten_vec_postops(
            program.name,
            program.to_native(),
            A,
            B,
            tensors,
        )

        return out, side_outputs


class OneDnnX64BrgemmProvider(CpuGemmProvider):
    name = "onednn-x64-brgemm"

    @classmethod
    def is_available(cls, features: CpuBackendFeatures) -> bool:
        native = load_native_extension()
        architecture_supported = (
            features.machine in _X86_64_MACHINES and features.has_avx2
        ) or (
            features.machine in _AARCH64_MACHINES and features.has_sve
        )
        return (
            native is not None
            and architecture_supported
            and bool(native.has_onednn())
        )

    def execute(
        self,
        program: GemmEpilogueProgram,
        A: torch.Tensor,
        B: torch.Tensor,
        **tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        native = load_native_extension()
        if native is None or not bool(native.has_onednn()):
            return super().execute(program, A, B, **tensors)
        return native.execute_brgemm_postops(
            program.name,
            program.to_native(),
            A,
            B,
            tensors,
        )


class LibxsmmProvider(CpuGemmProvider):
    name = "libxsmm"

    @classmethod
    def is_available(cls, features: CpuBackendFeatures) -> bool:
        native = load_native_extension()
        architecture_supported = (
            features.machine in _X86_64_MACHINES and features.has_avx2
        ) or (
            features.machine in _AARCH64_MACHINES
            and (features.has_neon or features.has_sve)
        )
        return (
            native is not None
            and architecture_supported
            and hasattr(native, "has_libxsmm")
            and bool(native.has_libxsmm())
        )

    def execute(
        self,
        program: GemmEpilogueProgram,
        A: torch.Tensor,
        B: torch.Tensor,
        **tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        native = load_native_extension()
        if native is None or not hasattr(native, "has_libxsmm") or not bool(native.has_libxsmm()):
            return super().execute(program, A, B, **tensors)
        return native.execute_libxsmm_postops(
            program.name,
            program.to_native(),
            A,
            B,
            tensors,
        )


_PROVIDER_CLASSES: tuple[type[CpuGemmProvider], ...] = (
    AtenFallbackProvider,
    AtenVecProvider,
    OneDnnX64BrgemmProvider,
    LibxsmmProvider,
)


@lru_cache(maxsize=None)
def _select_provider_cached(requested: str | None) -> CpuGemmProvider:
    features = detect_features()

    if requested:
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


def select_provider(preferred: str | None = None) -> CpuGemmProvider:
    requested = preferred or os.environ.get("CODA_CPU_PROVIDER")
    return _select_provider_cached(requested.lower() if requested else None)


def current_provider_name() -> str:
    return select_provider().name
