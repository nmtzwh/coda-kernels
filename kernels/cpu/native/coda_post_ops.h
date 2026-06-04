#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/extension.h>

namespace coda::cpu {

enum class PostOpKind : int32_t {
    ResidualAdd,
    RowBias,
    ColBias,
    ScalarScale,
    RowScale,
    ColScale,
    RowVectorScaleSideOutput,
    BlockMeanSquareReduction,
    RowSum,
    RowSumSquares,
    BlockMax,
    Silu,
    GeluTanh,
    Relu,
    SwiGLU,
    RoPE,
    TargetLogitSelect,
    BlockLogSumExp,
    StoreAccumulator,
};

struct PostOpNode {
    PostOpKind kind;
    std::string output;
    std::string tensor;
    int64_t block_size = 0;
    double value = 0.0;
    bool backward = false;
};

using TensorMap = std::unordered_map<std::string, at::Tensor>;

std::vector<PostOpNode> parse_post_ops(const pybind11::list &nodes);
TensorMap parse_tensor_map(const pybind11::dict &tensors);
std::pair<at::Tensor, pybind11::dict> run_post_ops(
        at::Tensor acc,
        const std::vector<PostOpNode> &nodes,
        const TensorMap &tensors,
        at::ScalarType output_dtype);

bool has_onednn();
bool has_libxsmm();
bool has_aten_vec();
std::string aten_vec_isa();

std::pair<at::Tensor, pybind11::dict> execute_brgemm_postops(
        const std::string &program_name,
        const pybind11::list &nodes,
        const at::Tensor &A,
        const at::Tensor &B,
        const pybind11::dict &tensors);

std::pair<at::Tensor, pybind11::dict> execute_libxsmm_postops(
        const std::string &program_name,
        const pybind11::list &nodes,
        const at::Tensor &A,
        const at::Tensor &B,
        const pybind11::dict &tensors);

std::pair<at::Tensor, pybind11::dict> execute_aten_vec_postops(
        const std::string &program_name,
        const pybind11::list &nodes,
        const at::Tensor &A,
        const at::Tensor &B,
        const pybind11::dict &tensors);

}  // namespace coda::cpu
