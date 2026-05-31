from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class EpilogueOp(str, Enum):
    RESIDUAL_ADD = "residual_add"
    ROW_SCALE = "row_scale"
    ROW_VECTOR_SCALE_SIDE_OUTPUT = "row_vector_scale_side_output"
    BLOCK_MEAN_SQUARE_REDUCTION = "block_mean_square_reduction"
    SWIGLU = "swiglu"
    ROPE = "rope"
    TARGET_LOGIT_SELECT = "target_logit_select"
    BLOCK_LOGSUMEXP = "block_logsumexp"


@dataclass(frozen=True)
class EpilogueNode:
    op: EpilogueOp
    output: str | None = None
    tensor: str | None = None
    block_size: int | None = None
    backward: bool = False


@dataclass(frozen=True)
class GemmEpilogueProgram:
    name: str
    nodes: tuple[EpilogueNode, ...]
    output_dtype: Any | None = None


def residual_add(tensor: str = "C") -> EpilogueNode:
    return EpilogueNode(op=EpilogueOp.RESIDUAL_ADD, tensor=tensor)


def row_scale(tensor: str = "R") -> EpilogueNode:
    return EpilogueNode(op=EpilogueOp.ROW_SCALE, tensor=tensor)


def row_vector_scale_side_output(output: str, tensor: str = "W") -> EpilogueNode:
    return EpilogueNode(
        op=EpilogueOp.ROW_VECTOR_SCALE_SIDE_OUTPUT,
        output=output,
        tensor=tensor,
    )


def block_mean_square_reduction(output: str, block_size: int) -> EpilogueNode:
    return EpilogueNode(
        op=EpilogueOp.BLOCK_MEAN_SQUARE_REDUCTION,
        output=output,
        block_size=block_size,
    )


def swiglu(output: str = "O") -> EpilogueNode:
    return EpilogueNode(op=EpilogueOp.SWIGLU, output=output)


def rope(output: str = "O", tensor: str = "cos_sin", backward: bool = False) -> EpilogueNode:
    return EpilogueNode(
        op=EpilogueOp.ROPE,
        output=output,
        tensor=tensor,
        backward=backward,
    )


def target_logit_select(output: str = "logits_tgt", tensor: str = "targets") -> EpilogueNode:
    return EpilogueNode(
        op=EpilogueOp.TARGET_LOGIT_SELECT,
        output=output,
        tensor=tensor,
    )


def block_logsumexp(output: str, block_size: int) -> EpilogueNode:
    return EpilogueNode(
        op=EpilogueOp.BLOCK_LOGSUMEXP,
        output=output,
        block_size=block_size,
    )
