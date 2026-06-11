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
bool aten_vec_bf16_dot();
void prepack_weight(const at::Tensor &B);
at::Tensor execute_aten_vec_gemm(const at::Tensor &A, const at::Tensor &B);
at::Tensor execute_aten_vec_rmsnorm(
        const at::Tensor &x,
        const at::Tensor &w,
        double eps);
at::Tensor execute_aten_vec_split_transpose_rope_cache(
        const at::Tensor &qkv,
        const at::Tensor &cos,
        const at::Tensor &sin,
        at::Tensor &k_cache,
        at::Tensor &v_cache,
        int64_t layer_idx,
        int64_t cache_index,
        int64_t num_heads,
        int64_t num_kv_heads,
        int64_t head_dim,
        bool is_qwen3,
        const at::Tensor &q_norm_w,
        const at::Tensor &k_norm_w,
        double rms_norm_eps);
at::Tensor execute_aten_vec_decode_attention(
        const at::Tensor &Q,
        const at::Tensor &k_cache,
        const at::Tensor &v_cache,
        int64_t layer_idx,
        int64_t seq_len,
        int64_t num_heads,
        int64_t num_kv_heads,
        int64_t head_dim);

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

class CodaQwenModel {
public:
    at::Tensor embed_tokens_weight;
    std::vector<at::Tensor> input_layernorm_weights;
    std::vector<at::Tensor> post_attention_layernorm_weights;
    std::vector<at::Tensor> w3_weights;
    std::vector<at::Tensor> qkv_biases;
    std::vector<at::Tensor> w0_weights;
    std::vector<at::Tensor> w1_weights;
    std::vector<at::Tensor> w2_weights;
    std::vector<at::Tensor> q_norm_weights;
    std::vector<at::Tensor> k_norm_weights;
    at::Tensor final_norm_weight;
    at::Tensor lm_head_weight;
    double rms_norm_eps;
    int64_t num_heads;
    int64_t num_kv_heads;
    int64_t head_dim;
    bool is_qwen3;
    int64_t num_layers;

    CodaQwenModel(
        at::Tensor embed_tokens_weight,
        std::vector<at::Tensor> input_layernorm_weights,
        std::vector<at::Tensor> post_attention_layernorm_weights,
        std::vector<at::Tensor> w3_weights,
        std::vector<at::Tensor> qkv_biases,
        std::vector<at::Tensor> w0_weights,
        std::vector<at::Tensor> w1_weights,
        std::vector<at::Tensor> w2_weights,
        std::vector<at::Tensor> q_norm_weights,
        std::vector<at::Tensor> k_norm_weights,
        at::Tensor final_norm_weight,
        at::Tensor lm_head_weight,
        double rms_norm_eps,
        int64_t num_heads,
        int64_t num_kv_heads,
        int64_t head_dim,
        bool is_qwen3
    );

    at::Tensor forward(
        const at::Tensor &input_ids,
        const at::Tensor &cos,
        const at::Tensor &sin,
        at::Tensor &k_cache,
        at::Tensor &v_cache,
        int64_t cache_index
    );
};

}  // namespace coda::cpu
