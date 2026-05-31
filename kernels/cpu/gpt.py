from __future__ import annotations

import torch

from kernels.cpu.ir import (
    GemmEpilogueProgram,
    block_logsumexp,
    block_mean_square_reduction,
    residual_add,
    rope,
    row_scale,
    row_vector_scale_side_output,
    swiglu,
    target_logit_select,
)
from kernels.cpu.providers import select_provider


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _require_cpu(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if tensor.device.type != "cpu":
            raise ValueError(f"backend='cpu' expects CPU tensors, got {tensor.device}")


def _execute(
    program: GemmEpilogueProgram,
    A: torch.Tensor,
    B: torch.Tensor,
    **kwargs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _require_cpu(A, B, *kwargs.values())
    return select_provider().execute(program, A, B, **kwargs)


def gemm_residual_partial_rmsnorm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    W: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, _ = A.shape
    _, N = B.shape
    if N % block_size != 0:
        raise ValueError(f"N={N} must be divisible by block_size={block_size}")
    program = GemmEpilogueProgram(
        name="gemm_residual_partial_rmsnorm",
        nodes=(
            residual_add("C"),
            block_mean_square_reduction("S", block_size),
            row_vector_scale_side_output("O", "W"),
        ),
        output_dtype=A.dtype,
    )
    D, outputs = _execute(program, A, B, C=C, W=W)
    S = outputs["S"].to(dtype=torch.float32)
    O = outputs["O"].to(dtype=A.dtype)
    if S.shape != (M, _ceil_div(N, block_size)):
        raise RuntimeError(f"unexpected S shape: {tuple(S.shape)}")
    return D, S, O


def gemm_rmsnorm(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    program = GemmEpilogueProgram(
        name="gemm_rmsnorm",
        nodes=(row_scale("R"),),
        output_dtype=A.dtype,
    )
    D, _ = _execute(program, A, B, R=R)
    return D


def gemm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, N = B.shape
    if N % 2 != 0:
        raise ValueError(f"SwiGLU expects even N, got {N}")
    program = GemmEpilogueProgram(
        name="gemm_swiglu",
        nodes=(swiglu("O"),),
        output_dtype=A.dtype,
    )
    D, outputs = _execute(program, A, B)
    return D, outputs["O"].to(dtype=A.dtype)


def gemm_rmsnorm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, N = B.shape
    if N % 2 != 0:
        raise ValueError(f"SwiGLU expects even N, got {N}")
    program = GemmEpilogueProgram(
        name="gemm_rmsnorm_swiglu",
        nodes=(row_scale("R"), swiglu("O")),
        output_dtype=A.dtype,
    )
    D, outputs = _execute(program, A, B, R=R)
    return D, outputs["O"].to(dtype=A.dtype)


def gemm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    cos_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    program = GemmEpilogueProgram(
        name="gemm_rope",
        nodes=(rope("O", "cos_sin"),),
        output_dtype=A.dtype,
    )
    D, outputs = _execute(program, A, B, cos_sin=cos_sin)
    return D, outputs["O"].to(dtype=A.dtype)


def gemm_rmsnorm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    cos_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    program = GemmEpilogueProgram(
        name="gemm_rmsnorm_rope",
        nodes=(row_scale("R"), rope("O", "cos_sin")),
        output_dtype=A.dtype,
    )
    D, outputs = _execute(program, A, B, R=R, cos_sin=cos_sin)
    return D, outputs["O"].to(dtype=A.dtype)


def gemm_partial_cross_entropy(
    A: torch.Tensor,
    B: torch.Tensor,
    targets: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, N = B.shape
    if N % block_size != 0:
        raise ValueError(f"N={N} must be divisible by block_size={block_size}")
    program = GemmEpilogueProgram(
        name="gemm_partial_cross_entropy",
        nodes=(
            target_logit_select("logits_tgt", "targets"),
            block_logsumexp("logits_lse", block_size),
        ),
        output_dtype=A.dtype,
    )
    logits, outputs = _execute(program, A, B, targets=targets)
    return (
        logits,
        outputs["logits_tgt"].to(dtype=A.dtype),
        outputs["logits_lse"].to(dtype=torch.float32),
    )


def gemm_rmsnorm_partial_cross_entropy(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    targets: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, N = B.shape
    if N % block_size != 0:
        raise ValueError(f"N={N} must be divisible by block_size={block_size}")
    program = GemmEpilogueProgram(
        name="gemm_rmsnorm_partial_cross_entropy",
        nodes=(
            row_scale("R"),
            target_logit_select("logits_tgt", "targets"),
            block_logsumexp("logits_lse", block_size),
        ),
        output_dtype=A.dtype,
    )
    logits, outputs = _execute(program, A, B, R=R, targets=targets)
    return (
        logits,
        outputs["logits_tgt"].to(dtype=A.dtype),
        outputs["logits_lse"].to(dtype=torch.float32),
    )


def gemm_residual_partial_rmsnorm_bwd(*args, **kwargs):
    raise NotImplementedError("backend='cpu' currently implements forward inference only")


def gemm_partial_swiglu_bwd(*args, **kwargs):
    raise NotImplementedError("backend='cpu' currently implements forward inference only")
