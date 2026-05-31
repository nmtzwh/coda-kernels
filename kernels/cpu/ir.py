from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class EpilogueOp(str, Enum):
    RESIDUAL_ADD = "residual_add"
    ROW_BIAS = "row_bias"
    COL_BIAS = "col_bias"
    SCALAR_SCALE = "scalar_scale"
    ROW_SCALE = "row_scale"
    COL_SCALE = "col_scale"
    ROW_VECTOR_SCALE_SIDE_OUTPUT = "row_vector_scale_side_output"
    BLOCK_MEAN_SQUARE_REDUCTION = "block_mean_square_reduction"
    ROW_SUM = "row_sum"
    ROW_SUM_SQUARES = "row_sum_squares"
    BLOCK_MAX = "block_max"
    SILU = "silu"
    GELU_TANH = "gelu_tanh"
    RELU = "relu"
    SWIGLU = "swiglu"
    ROPE = "rope"
    TARGET_LOGIT_SELECT = "target_logit_select"
    BLOCK_LOGSUMEXP = "block_logsumexp"
    STORE_ACCUMULATOR = "store_accumulator"


@dataclass(frozen=True)
class EpilogueNode:
    op: EpilogueOp
    output: str | None = None
    tensor: str | None = None
    block_size: int | None = None
    value: float | None = None
    backward: bool = False

    def to_native(self) -> dict:
        return {
            "op": self.op.value,
            "output": self.output,
            "tensor": self.tensor,
            "block_size": self.block_size,
            "value": self.value,
            "backward": self.backward,
        }


@dataclass(frozen=True)
class GemmEpilogueProgram:
    name: str
    nodes: tuple[EpilogueNode, ...]
    output_dtype: Any | None = None

    def to_native(self) -> list[dict]:
        return [node.to_native() for node in self.nodes]


def residual_add(tensor: str = "C") -> EpilogueNode:
    return EpilogueNode(op=EpilogueOp.RESIDUAL_ADD, tensor=tensor)


def row_bias(tensor: str) -> EpilogueNode:
    return EpilogueNode(op=EpilogueOp.ROW_BIAS, tensor=tensor)


def col_bias(tensor: str) -> EpilogueNode:
    return EpilogueNode(op=EpilogueOp.COL_BIAS, tensor=tensor)


def scalar_scale(value: float) -> EpilogueNode:
    return EpilogueNode(op=EpilogueOp.SCALAR_SCALE, value=value)


def row_scale(tensor: str = "R") -> EpilogueNode:
    return EpilogueNode(op=EpilogueOp.ROW_SCALE, tensor=tensor)


def col_scale(tensor: str) -> EpilogueNode:
    return EpilogueNode(op=EpilogueOp.COL_SCALE, tensor=tensor)


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


def row_sum(output: str) -> EpilogueNode:
    return EpilogueNode(op=EpilogueOp.ROW_SUM, output=output)


def row_sum_squares(output: str) -> EpilogueNode:
    return EpilogueNode(op=EpilogueOp.ROW_SUM_SQUARES, output=output)


def block_max(output: str, block_size: int) -> EpilogueNode:
    return EpilogueNode(
        op=EpilogueOp.BLOCK_MAX,
        output=output,
        block_size=block_size,
    )


def silu() -> EpilogueNode:
    return EpilogueNode(op=EpilogueOp.SILU)


def gelu_tanh() -> EpilogueNode:
    return EpilogueNode(op=EpilogueOp.GELU_TANH)


def relu() -> EpilogueNode:
    return EpilogueNode(op=EpilogueOp.RELU)


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


def store_accumulator(output: str) -> EpilogueNode:
    return EpilogueNode(op=EpilogueOp.STORE_ACCUMULATOR, output=output)
