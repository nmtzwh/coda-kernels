#include "coda_post_ops.h"

#include <ATen/Parallel.h>

#include <cmath>
#include <stdexcept>

namespace py = pybind11;

namespace coda::cpu {
namespace {

int64_t ceil_div(int64_t x, int64_t y) {
    return (x + y - 1) / y;
}

PostOpKind parse_kind(const std::string &op) {
    if (op == "residual_add") return PostOpKind::ResidualAdd;
    if (op == "row_bias") return PostOpKind::RowBias;
    if (op == "col_bias") return PostOpKind::ColBias;
    if (op == "scalar_scale") return PostOpKind::ScalarScale;
    if (op == "row_scale") return PostOpKind::RowScale;
    if (op == "col_scale") return PostOpKind::ColScale;
    if (op == "row_vector_scale_side_output") return PostOpKind::RowVectorScaleSideOutput;
    if (op == "block_mean_square_reduction") return PostOpKind::BlockMeanSquareReduction;
    if (op == "row_sum") return PostOpKind::RowSum;
    if (op == "row_sum_squares") return PostOpKind::RowSumSquares;
    if (op == "block_max") return PostOpKind::BlockMax;
    if (op == "silu") return PostOpKind::Silu;
    if (op == "gelu_tanh") return PostOpKind::GeluTanh;
    if (op == "relu") return PostOpKind::Relu;
    if (op == "swiglu") return PostOpKind::SwiGLU;
    if (op == "rope") return PostOpKind::RoPE;
    if (op == "target_logit_select") return PostOpKind::TargetLogitSelect;
    if (op == "block_logsumexp") return PostOpKind::BlockLogSumExp;
    if (op == "store_accumulator") return PostOpKind::StoreAccumulator;
    throw std::invalid_argument("unknown CODA post-op: " + op);
}

std::string get_optional_string(const py::dict &node, const char *key) {
    if (!node.contains(key) || node[key].is_none()) return "";
    return py::cast<std::string>(node[key]);
}

int64_t get_optional_i64(const py::dict &node, const char *key) {
    if (!node.contains(key) || node[key].is_none()) return 0;
    return py::cast<int64_t>(node[key]);
}

double get_optional_f64(const py::dict &node, const char *key) {
    if (!node.contains(key) || node[key].is_none()) return 0.0;
    return py::cast<double>(node[key]);
}

bool get_optional_bool(const py::dict &node, const char *key) {
    if (!node.contains(key) || node[key].is_none()) return false;
    return py::cast<bool>(node[key]);
}

at::Tensor to_acc(const at::Tensor &x) {
    return x.to(at::kFloat);
}

at::Tensor row_view(const at::Tensor &x) {
    return to_acc(x).reshape({-1, 1});
}

at::Tensor col_view(const at::Tensor &x) {
    return to_acc(x).reshape({1, -1});
}

const at::Tensor &require_tensor(const TensorMap &tensors, const std::string &name) {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        throw std::invalid_argument("missing CODA post-op tensor: " + name);
    }
    return it->second;
}

std::string require_name(const std::string &name, const char *field) {
    if (name.empty()) {
        throw std::invalid_argument(std::string("missing CODA post-op ") + field);
    }
    return name;
}

at::Tensor swiglu(const at::Tensor &acc) {
    const int64_t n = acc.size(-1);
    if (n % 2 != 0) {
        throw std::invalid_argument("SwiGLU expects an even last dimension");
    }
    auto x = acc.reshape({acc.size(0), n / 2, 2});
    auto gate = x.select(-1, 0);
    auto up = x.select(-1, 1);
    return at::silu(gate) * up;
}

at::Tensor rope(const at::Tensor &acc, const at::Tensor &cos_sin, bool backward) {
    if (!acc.sizes().equals(cos_sin.sizes())) {
        throw std::invalid_argument("RoPE cos_sin shape must match accumulator shape");
    }
    const int64_t n = acc.size(-1);
    if (n % 2 != 0) {
        throw std::invalid_argument("RoPE expects an even last dimension");
    }
    const double sign = backward ? -1.0 : 1.0;
    auto x = acc.reshape({acc.size(0), n / 2, 2});
    auto cs = to_acc(cos_sin).reshape({cos_sin.size(0), n / 2, 2});
    auto x0 = x.select(-1, 0);
    auto x1 = x.select(-1, 1);
    auto c = cs.select(-1, 0);
    auto s = cs.select(-1, 1) * sign;
    auto y0 = x0 * c + x1 * s;
    auto y1 = x0 * (-s) + x1 * c;
    return at::stack({y0, y1}, -1).reshape_as(acc);
}

at::Tensor block_view(const at::Tensor &acc, int64_t block_size) {
    if (block_size <= 0) {
        throw std::invalid_argument("block post-op requires a positive block_size");
    }
    if (acc.size(1) % block_size != 0) {
        throw std::invalid_argument("N must be divisible by block_size");
    }
    return acc.reshape({acc.size(0), ceil_div(acc.size(1), block_size), block_size});
}

at::Tensor output_tensor(const at::Tensor &acc, at::ScalarType output_dtype) {
    return acc.scalar_type() == output_dtype ? acc : acc.to(output_dtype);
}

void require_contiguous_matrix(const at::Tensor &x, const char *name) {
    TORCH_CHECK(x.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(x.dim() == 2, name, " must be a 2D tensor");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kFloat, name, " must be float32");
}

void require_contiguous_vector(const at::Tensor &x, int64_t size, const char *name) {
    TORCH_CHECK(x.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(x.dim() == 1, name, " must be a 1D tensor");
    TORCH_CHECK(x.size(0) == size, name, " has an unexpected shape");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kFloat, name, " must be float32");
}

float silu_scalar(float x) {
    return x / (1.0f + std::exp(-x));
}

void row_scale_inplace(at::Tensor &acc, const at::Tensor &R) {
    require_contiguous_matrix(acc, "acc");
    require_contiguous_vector(R, acc.size(0), "R");

    const int64_t M = acc.size(0);
    const int64_t N = acc.size(1);
    float *acc_ptr = acc.data_ptr<float>();
    const float *r_ptr = R.data_ptr<float>();

    at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
        for (int64_t m = begin; m < end; ++m) {
            const float scale = r_ptr[m];
            float *row = acc_ptr + m * N;
            for (int64_t n = 0; n < N; ++n) {
                row[n] *= scale;
            }
        }
    });
}

at::Tensor row_scale_swiglu_inplace(at::Tensor &acc, const at::Tensor &R) {
    require_contiguous_matrix(acc, "acc");
    require_contiguous_vector(R, acc.size(0), "R");
    TORCH_CHECK(acc.size(1) % 2 == 0, "SwiGLU expects an even last dimension");

    const int64_t M = acc.size(0);
    const int64_t N = acc.size(1);
    auto O = at::empty({M, N / 2}, acc.options().dtype(at::kFloat));
    float *acc_ptr = acc.data_ptr<float>();
    float *out_ptr = O.data_ptr<float>();
    const float *r_ptr = R.data_ptr<float>();

    at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
        for (int64_t m = begin; m < end; ++m) {
            const float scale = r_ptr[m];
            float *row = acc_ptr + m * N;
            float *out_row = out_ptr + m * (N / 2);
            for (int64_t n = 0, j = 0; n < N; n += 2, ++j) {
                const float gate = row[n] * scale;
                const float up = row[n + 1] * scale;
                row[n] = gate;
                row[n + 1] = up;
                out_row[j] = silu_scalar(gate) * up;
            }
        }
    });

    return O;
}

at::Tensor row_scale_rope_inplace(
        at::Tensor &acc,
        const at::Tensor &R,
        const at::Tensor &cos_sin,
        bool backward) {
    require_contiguous_matrix(acc, "acc");
    require_contiguous_vector(R, acc.size(0), "R");
    require_contiguous_matrix(cos_sin, "cos_sin");
    TORCH_CHECK(cos_sin.sizes().equals(acc.sizes()), "RoPE cos_sin shape must match accumulator shape");
    TORCH_CHECK(acc.size(1) % 2 == 0, "RoPE expects an even last dimension");

    const int64_t M = acc.size(0);
    const int64_t N = acc.size(1);
    const float sign = backward ? -1.0f : 1.0f;
    auto O = at::empty({M, N}, acc.options().dtype(at::kFloat));
    float *acc_ptr = acc.data_ptr<float>();
    float *out_ptr = O.data_ptr<float>();
    const float *r_ptr = R.data_ptr<float>();
    const float *cs_ptr = cos_sin.data_ptr<float>();

    at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
        for (int64_t m = begin; m < end; ++m) {
            const float scale = r_ptr[m];
            float *row = acc_ptr + m * N;
            float *out_row = out_ptr + m * N;
            const float *cs_row = cs_ptr + m * N;
            for (int64_t n = 0; n < N; n += 2) {
                const float x0 = row[n] * scale;
                const float x1 = row[n + 1] * scale;
                const float c = cs_row[n];
                const float s = cs_row[n + 1] * sign;
                row[n] = x0;
                row[n + 1] = x1;
                out_row[n] = x0 * c + x1 * s;
                out_row[n + 1] = x0 * (-s) + x1 * c;
            }
        }
    });

    return O;
}

std::pair<at::Tensor, at::Tensor> residual_partial_rmsnorm_inplace(
        at::Tensor &acc,
        const at::Tensor &C,
        const at::Tensor &W,
        int64_t block_size) {
    require_contiguous_matrix(acc, "acc");
    require_contiguous_matrix(C, "C");
    require_contiguous_vector(W, acc.size(1), "W");
    TORCH_CHECK(C.sizes().equals(acc.sizes()), "residual shape mismatch");
    TORCH_CHECK(block_size > 0, "block post-op requires a positive block_size");
    TORCH_CHECK(acc.size(1) % block_size == 0, "N must be divisible by block_size");

    const int64_t M = acc.size(0);
    const int64_t N = acc.size(1);
    const int64_t num_blocks = N / block_size;
    auto S = at::empty({M, num_blocks}, acc.options().dtype(at::kFloat));
    auto O = at::empty({M, N}, acc.options().dtype(at::kFloat));
    float *acc_ptr = acc.data_ptr<float>();
    float *s_ptr = S.data_ptr<float>();
    float *out_ptr = O.data_ptr<float>();
    const float *c_ptr = C.data_ptr<float>();
    const float *w_ptr = W.data_ptr<float>();

    at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
        for (int64_t m = begin; m < end; ++m) {
            float *row = acc_ptr + m * N;
            const float *c_row = c_ptr + m * N;
            float *out_row = out_ptr + m * N;
            float *s_row = s_ptr + m * num_blocks;
            for (int64_t block = 0; block < num_blocks; ++block) {
                const int64_t n0 = block * block_size;
                float sum_sq = 0.0f;
                for (int64_t offset = 0; offset < block_size; ++offset) {
                    const int64_t n = n0 + offset;
                    const float value = row[n] + c_row[n];
                    row[n] = value;
                    sum_sq += value * value;
                    out_row[n] = value * w_ptr[n];
                }
                s_row[block] = sum_sq / static_cast<float>(block_size);
            }
        }
    });

    return {S, O};
}

}  // namespace

std::vector<PostOpNode> parse_post_ops(const py::list &nodes) {
    std::vector<PostOpNode> parsed;
    parsed.reserve(nodes.size());
    for (const auto &item : nodes) {
        auto node = py::cast<py::dict>(item);
        PostOpNode parsed_node;
        parsed_node.kind = parse_kind(py::cast<std::string>(node["op"]));
        parsed_node.output = get_optional_string(node, "output");
        parsed_node.tensor = get_optional_string(node, "tensor");
        parsed_node.block_size = get_optional_i64(node, "block_size");
        parsed_node.value = get_optional_f64(node, "value");
        parsed_node.backward = get_optional_bool(node, "backward");
        parsed.push_back(std::move(parsed_node));
    }
    return parsed;
}

TensorMap parse_tensor_map(const py::dict &tensors) {
    TensorMap result;
    for (auto item : tensors) {
        result.emplace(py::cast<std::string>(item.first), py::cast<at::Tensor>(item.second));
    }
    return result;
}

std::pair<at::Tensor, py::dict> run_post_ops(
        at::Tensor acc,
        const std::vector<PostOpNode> &nodes,
        const TensorMap &tensors,
        at::ScalarType output_dtype) {
    acc = to_acc(acc);
    if (!acc.is_contiguous()) {
        acc = acc.contiguous();
    }

    if (nodes.size() == 1 && nodes[0].kind == PostOpKind::RowScale) {
        const auto R = to_acc(require_tensor(tensors, require_name(nodes[0].tensor, "tensor"))).contiguous();
        row_scale_inplace(acc, R);
        return {output_tensor(acc, output_dtype), py::dict()};
    }

    if (nodes.size() == 2 &&
            nodes[0].kind == PostOpKind::RowScale &&
            nodes[1].kind == PostOpKind::SwiGLU) {
        const auto R = to_acc(require_tensor(tensors, require_name(nodes[0].tensor, "tensor"))).contiguous();
        auto O = row_scale_swiglu_inplace(acc, R);
        py::dict side_outputs;
        side_outputs[py::str(require_name(nodes[1].output, "output"))] = O;
        return {output_tensor(acc, output_dtype), side_outputs};
    }

    if (nodes.size() == 2 &&
            nodes[0].kind == PostOpKind::RowScale &&
            nodes[1].kind == PostOpKind::RoPE) {
        const auto R = to_acc(require_tensor(tensors, require_name(nodes[0].tensor, "tensor"))).contiguous();
        const auto cos_sin =
                to_acc(require_tensor(tensors, require_name(nodes[1].tensor, "tensor"))).contiguous();
        auto O = row_scale_rope_inplace(acc, R, cos_sin, nodes[1].backward);
        py::dict side_outputs;
        side_outputs[py::str(require_name(nodes[1].output, "output"))] = O;
        return {output_tensor(acc, output_dtype), side_outputs};
    }

    if (nodes.size() == 3 &&
            nodes[0].kind == PostOpKind::ResidualAdd &&
            nodes[1].kind == PostOpKind::BlockMeanSquareReduction &&
            nodes[2].kind == PostOpKind::RowVectorScaleSideOutput) {
        const auto C = to_acc(require_tensor(tensors, require_name(nodes[0].tensor, "tensor"))).contiguous();
        const auto W = to_acc(require_tensor(tensors, require_name(nodes[2].tensor, "tensor"))).contiguous();
        auto outputs = residual_partial_rmsnorm_inplace(acc, C, W, nodes[1].block_size);
        py::dict side_outputs;
        side_outputs[py::str(require_name(nodes[1].output, "output"))] = outputs.first;
        side_outputs[py::str(require_name(nodes[2].output, "output"))] = outputs.second;
        return {output_tensor(acc, output_dtype), side_outputs};
    }

    py::dict side_outputs;

    for (const auto &node : nodes) {
        switch (node.kind) {
            case PostOpKind::ResidualAdd:
                acc = acc + to_acc(require_tensor(tensors, require_name(node.tensor, "tensor")));
                break;
            case PostOpKind::RowBias:
                acc = acc + row_view(require_tensor(tensors, require_name(node.tensor, "tensor")));
                break;
            case PostOpKind::ColBias:
                acc = acc + col_view(require_tensor(tensors, require_name(node.tensor, "tensor")));
                break;
            case PostOpKind::ScalarScale:
                acc = acc * node.value;
                break;
            case PostOpKind::RowScale:
                acc = acc * row_view(require_tensor(tensors, require_name(node.tensor, "tensor")));
                break;
            case PostOpKind::ColScale:
                acc = acc * col_view(require_tensor(tensors, require_name(node.tensor, "tensor")));
                break;
            case PostOpKind::RowVectorScaleSideOutput:
                side_outputs[py::str(require_name(node.output, "output"))] =
                        acc * col_view(require_tensor(tensors, require_name(node.tensor, "tensor")));
                break;
            case PostOpKind::BlockMeanSquareReduction:
                side_outputs[py::str(require_name(node.output, "output"))] =
                        block_view(acc, node.block_size).square().mean(-1);
                break;
            case PostOpKind::RowSum:
                side_outputs[py::str(require_name(node.output, "output"))] = acc.sum(-1);
                break;
            case PostOpKind::RowSumSquares:
                side_outputs[py::str(require_name(node.output, "output"))] = acc.square().sum(-1);
                break;
            case PostOpKind::BlockMax:
                side_outputs[py::str(require_name(node.output, "output"))] =
                        std::get<0>(block_view(acc, node.block_size).max(-1));
                break;
            case PostOpKind::Silu:
                acc = at::silu(acc);
                break;
            case PostOpKind::GeluTanh:
                acc = at::gelu(acc, "tanh");
                break;
            case PostOpKind::Relu:
                acc = at::relu(acc);
                break;
            case PostOpKind::SwiGLU:
                side_outputs[py::str(require_name(node.output, "output"))] = swiglu(acc);
                break;
            case PostOpKind::RoPE:
                side_outputs[py::str(require_name(node.output, "output"))] =
                        rope(acc, require_tensor(tensors, require_name(node.tensor, "tensor")), node.backward);
                break;
            case PostOpKind::TargetLogitSelect: {
                auto targets = require_tensor(tensors, require_name(node.tensor, "tensor")).to(at::kLong);
                auto rows = at::arange(acc.size(0), acc.options().dtype(at::kLong));
                side_outputs[py::str(require_name(node.output, "output"))] =
                        acc.index({rows, targets});
                break;
            }
            case PostOpKind::BlockLogSumExp:
                side_outputs[py::str(require_name(node.output, "output"))] =
                        at::logsumexp(block_view(acc, node.block_size), {-1});
                break;
            case PostOpKind::StoreAccumulator:
                side_outputs[py::str(require_name(node.output, "output"))] = acc;
                break;
        }
    }

    return {acc.to(output_dtype), side_outputs};
}

}  // namespace coda::cpu
