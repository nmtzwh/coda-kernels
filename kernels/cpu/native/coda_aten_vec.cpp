#include "coda_post_ops.h"

#include <ATen/Parallel.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>

#if defined(CODA_CPU_WITH_ATEN_VEC)
#include <ATen/cpu/vec/vec.h>
#endif

namespace py = pybind11;

namespace coda::cpu {
namespace {

#if defined(CODA_CPU_WITH_ATEN_VEC)
using Vec = at::vec::Vectorized<float>;
#if !defined(CODA_CPU_ATEN_VEC_L1_BYTES)
#define CODA_CPU_ATEN_VEC_L1_BYTES 32768
#endif
#if !defined(CODA_CPU_ATEN_VEC_L2_BYTES)
#define CODA_CPU_ATEN_VEC_L2_BYTES 524288
#endif
constexpr int64_t kMaxMcRows = 64;
constexpr int64_t kMaxBlockN = 3 * Vec::size();
constexpr int64_t kMaxAccBufferSize = kMaxMcRows * kMaxBlockN;
constexpr int64_t kL1Budget = CODA_CPU_ATEN_VEC_L1_BYTES * 4 / 5;
constexpr int64_t kL2Budget = CODA_CPU_ATEN_VEC_L2_BYTES * 3 / 4;

struct PackedBKey {
    const void *impl;
    const void *data;
    int64_t k;
    int64_t n;
    int64_t stride0;
    int64_t stride1;
    int64_t block_n;
    uint32_t version;

    bool operator==(const PackedBKey &other) const {
        return impl == other.impl &&
                data == other.data &&
                k == other.k &&
                n == other.n &&
                stride0 == other.stride0 &&
                stride1 == other.stride1 &&
                block_n == other.block_n &&
                version == other.version;
    }
};

struct PackedBKeyHash {
    size_t operator()(const PackedBKey &key) const {
        size_t h = std::hash<const void *>{}(key.impl);
        h ^= std::hash<const void *>{}(key.data) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.k) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.n) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.stride0) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.stride1) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.block_n) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>{}(key.version) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

struct PackedBEntry {
    at::Tensor source;
    at::Tensor packed;
};

std::mutex packed_b_cache_mutex;
std::unordered_map<PackedBKey, PackedBEntry, PackedBKeyHash> packed_b_cache;

PackedBKey packed_b_key(const at::Tensor &B, int64_t block_n) {
    return PackedBKey{
            B.unsafeGetTensorImpl(),
            B.data_ptr<float>(),
            B.size(0),
            B.size(1),
            B.stride(0),
            B.stride(1),
            block_n,
            B.unsafeGetTensorImpl()->version_counter().current_version(),
    };
}

int64_t ceil_div(int64_t x, int64_t y) {
    return (x + y - 1) / y;
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

void require_matrix(const at::Tensor &x, const char *name) {
    TORCH_CHECK(x.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(x.dim() == 2, name, " must be a 2D tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

void require_vector(const at::Tensor &x, int64_t size, const char *name) {
    TORCH_CHECK(x.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(x.dim() == 1, name, " must be a 1D tensor");
    TORCH_CHECK(x.size(0) == size, name, " has an unexpected shape");
    TORCH_CHECK(x.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

void require_targets(const at::Tensor &x, int64_t size, const char *name) {
    TORCH_CHECK(x.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(x.dim() == 1, name, " must be a 1D tensor");
    TORCH_CHECK(x.size(0) == size, name, " has an unexpected shape");
    TORCH_CHECK(x.scalar_type() == at::kLong, name, " must be int64");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

void require_gemm_inputs(const at::Tensor &A, const at::Tensor &B) {
    require_matrix(A, "A");
    require_matrix(B, "B");
    TORCH_CHECK(A.size(1) == B.size(0), "incompatible GEMM shapes");
    TORCH_CHECK(A.size(1) > 0, "GEMM K dimension must be positive");
}

void update_logsumexp(float value, float &max_value, float &sum_exp) {
    if (value > max_value) {
        sum_exp = sum_exp * std::exp(max_value - value) + 1.0f;
        max_value = value;
    } else {
        sum_exp += std::exp(value - max_value);
    }
}

template <int64_t col_vectors>
at::Tensor pack_b_register_blocks(const at::Tensor &B) {
    const int64_t K = B.size(0);
    const int64_t N = B.size(1);
    constexpr int64_t vec_size = Vec::size();
    constexpr int64_t block_n = col_vectors * vec_size;
    const int64_t num_blocks = ceil_div(N, block_n);
    const auto key = packed_b_key(B, block_n);
    {
        std::lock_guard<std::mutex> lock(packed_b_cache_mutex);
        auto it = packed_b_cache.find(key);
        if (it != packed_b_cache.end()) {
            return it->second.packed;
        }
    }

    auto packed = at::empty({num_blocks, K, block_n}, B.options().dtype(at::kFloat));
    const float *b_base = B.data_ptr<float>();
    float *packed_base = packed.data_ptr<float>();

    at::parallel_for(0, num_blocks, 1, [&](int64_t begin, int64_t end) {
        for (int64_t nb = begin; nb < end; ++nb) {
            for (int64_t k = 0; k < K; ++k) {
                float *packed_row = packed_base + (nb * K + k) * block_n;
                for (int64_t c = 0; c < col_vectors; ++c) {
                    const int64_t n = nb * block_n + c * vec_size;
                    const int64_t count = std::max<int64_t>(
                            0,
                            std::min<int64_t>(vec_size, N - n));
                    const Vec b_vec = count == 0
                            ? Vec(0.0f)
                            : Vec::loadu(b_base + k * N + n, count);
                    b_vec.store(packed_row + c * vec_size);
                }
            }
        }
    });

    {
        std::lock_guard<std::mutex> lock(packed_b_cache_mutex);
        if (packed_b_cache.size() >= 16) {
            packed_b_cache.clear();
        }
        packed_b_cache.emplace(key, PackedBEntry{B, packed});
    }

    return packed;
}

template <int64_t index, int64_t end, typename Fn>
inline void static_for(const Fn &fn) {
    if constexpr (index < end) {
        fn(std::integral_constant<int64_t, index>{});
        static_for<index + 1, end>(fn);
    }
}

template <bool accumulate>
inline Vec load_accumulator(const float *ptr) {
    if constexpr (accumulate) {
        return Vec::loadu(ptr);
    } else {
        return Vec(0.0f);
    }
}

template <bool accumulate>
inline void gemm_microkernel_4x2(
        const float *a_base,
        const float *b_panel,
        float *acc_base,
        int64_t K,
        int64_t k_begin,
        int64_t k_end) {
    constexpr int64_t vec_size = Vec::size();
    constexpr int64_t block_n = 2 * vec_size;
    Vec c00 = load_accumulator<accumulate>(acc_base);
    Vec c01 = load_accumulator<accumulate>(acc_base + vec_size);
    Vec c10 = load_accumulator<accumulate>(acc_base + block_n);
    Vec c11 = load_accumulator<accumulate>(acc_base + block_n + vec_size);
    Vec c20 = load_accumulator<accumulate>(acc_base + 2 * block_n);
    Vec c21 = load_accumulator<accumulate>(acc_base + 2 * block_n + vec_size);
    Vec c30 = load_accumulator<accumulate>(acc_base + 3 * block_n);
    Vec c31 = load_accumulator<accumulate>(acc_base + 3 * block_n + vec_size);

    for (int64_t k = k_begin; k < k_end; ++k) {
        const Vec b0 = Vec::loadu(b_panel + k * block_n);
        const Vec b1 = Vec::loadu(b_panel + k * block_n + vec_size);
        Vec a(a_base[k]);
        c00 = at::vec::fmadd(a, b0, c00);
        c01 = at::vec::fmadd(a, b1, c01);
        a = Vec(a_base[K + k]);
        c10 = at::vec::fmadd(a, b0, c10);
        c11 = at::vec::fmadd(a, b1, c11);
        a = Vec(a_base[2 * K + k]);
        c20 = at::vec::fmadd(a, b0, c20);
        c21 = at::vec::fmadd(a, b1, c21);
        a = Vec(a_base[3 * K + k]);
        c30 = at::vec::fmadd(a, b0, c30);
        c31 = at::vec::fmadd(a, b1, c31);
    }

    c00.store(acc_base);
    c01.store(acc_base + vec_size);
    c10.store(acc_base + block_n);
    c11.store(acc_base + block_n + vec_size);
    c20.store(acc_base + 2 * block_n);
    c21.store(acc_base + 2 * block_n + vec_size);
    c30.store(acc_base + 3 * block_n);
    c31.store(acc_base + 3 * block_n + vec_size);
}

template <bool accumulate>
inline void gemm_microkernel_4x3(
        const float *a_base,
        const float *b_panel,
        float *acc_base,
        int64_t K,
        int64_t k_begin,
        int64_t k_end) {
    constexpr int64_t vec_size = Vec::size();
    constexpr int64_t block_n = 3 * vec_size;
    Vec c00 = load_accumulator<accumulate>(acc_base);
    Vec c01 = load_accumulator<accumulate>(acc_base + vec_size);
    Vec c02 = load_accumulator<accumulate>(acc_base + 2 * vec_size);
    Vec c10 = load_accumulator<accumulate>(acc_base + block_n);
    Vec c11 = load_accumulator<accumulate>(acc_base + block_n + vec_size);
    Vec c12 = load_accumulator<accumulate>(acc_base + block_n + 2 * vec_size);
    Vec c20 = load_accumulator<accumulate>(acc_base + 2 * block_n);
    Vec c21 = load_accumulator<accumulate>(acc_base + 2 * block_n + vec_size);
    Vec c22 = load_accumulator<accumulate>(acc_base + 2 * block_n + 2 * vec_size);
    Vec c30 = load_accumulator<accumulate>(acc_base + 3 * block_n);
    Vec c31 = load_accumulator<accumulate>(acc_base + 3 * block_n + vec_size);
    Vec c32 = load_accumulator<accumulate>(acc_base + 3 * block_n + 2 * vec_size);

    for (int64_t k = k_begin; k < k_end; ++k) {
        const Vec b0 = Vec::loadu(b_panel + k * block_n);
        const Vec b1 = Vec::loadu(b_panel + k * block_n + vec_size);
        const Vec b2 = Vec::loadu(b_panel + k * block_n + 2 * vec_size);
        Vec a(a_base[k]);
        c00 = at::vec::fmadd(a, b0, c00);
        c01 = at::vec::fmadd(a, b1, c01);
        c02 = at::vec::fmadd(a, b2, c02);
        a = Vec(a_base[K + k]);
        c10 = at::vec::fmadd(a, b0, c10);
        c11 = at::vec::fmadd(a, b1, c11);
        c12 = at::vec::fmadd(a, b2, c12);
        a = Vec(a_base[2 * K + k]);
        c20 = at::vec::fmadd(a, b0, c20);
        c21 = at::vec::fmadd(a, b1, c21);
        c22 = at::vec::fmadd(a, b2, c22);
        a = Vec(a_base[3 * K + k]);
        c30 = at::vec::fmadd(a, b0, c30);
        c31 = at::vec::fmadd(a, b1, c31);
        c32 = at::vec::fmadd(a, b2, c32);
    }

    c00.store(acc_base);
    c01.store(acc_base + vec_size);
    c02.store(acc_base + 2 * vec_size);
    c10.store(acc_base + block_n);
    c11.store(acc_base + block_n + vec_size);
    c12.store(acc_base + block_n + 2 * vec_size);
    c20.store(acc_base + 2 * block_n);
    c21.store(acc_base + 2 * block_n + vec_size);
    c22.store(acc_base + 2 * block_n + 2 * vec_size);
    c30.store(acc_base + 3 * block_n);
    c31.store(acc_base + 3 * block_n + vec_size);
    c32.store(acc_base + 3 * block_n + 2 * vec_size);
}

template <int64_t rows, int64_t col_vectors, bool accumulate>
inline void gemm_microkernel(
        const float *a_base,
        const float *b_panel,
        float *acc_base,
        int64_t K,
        int64_t block_n,
        int64_t k_begin,
        int64_t k_end) {
    constexpr int64_t num_acc = rows * col_vectors;
    std::array<Vec, num_acc> acc;
    std::array<Vec, col_vectors> b_vecs;
    Vec a_vec;

    static_for<0, num_acc>([&](auto index) {
        constexpr int64_t row = index / col_vectors;
        constexpr int64_t col = index % col_vectors;
        if constexpr (accumulate) {
            acc[index] = Vec::loadu(acc_base + row * block_n + col * Vec::size());
        } else {
            acc[index] = Vec(0.0f);
        }
    });

    for (int64_t k = k_begin; k < k_end; ++k) {
        static_for<0, num_acc>([&](auto index) {
            constexpr int64_t row = index / col_vectors;
            constexpr int64_t col = index % col_vectors;
            if constexpr (col == 0) {
                a_vec = Vec(a_base[row * K + k]);
            }
            if constexpr (row == 0) {
                b_vecs[col] = Vec::loadu(b_panel + k * block_n + col * Vec::size());
            }
            acc[index] = at::vec::fmadd(a_vec, b_vecs[col], acc[index]);
        });
    }

    static_for<0, num_acc>([&](auto index) {
        constexpr int64_t row = index / col_vectors;
        constexpr int64_t col = index % col_vectors;
        acc[index].store(acc_base + row * block_n + col * Vec::size());
    });
}

template <int64_t col_vectors>
inline void gemm_microkernel_rows(
        int64_t rows,
        bool accumulate,
        const float *a_base,
        const float *b_panel,
        float *acc_base,
        int64_t K,
        int64_t block_n,
        int64_t k_begin,
        int64_t k_end) {
#define CODA_CALL_MICROKERNEL(ROWS) \
    if (accumulate) { \
        gemm_microkernel<ROWS, col_vectors, true>( \
                a_base, b_panel, acc_base, K, block_n, k_begin, k_end); \
    } else { \
        gemm_microkernel<ROWS, col_vectors, false>( \
                a_base, b_panel, acc_base, K, block_n, k_begin, k_end); \
    }

    switch (rows) {
        case 1:
            CODA_CALL_MICROKERNEL(1);
            break;
        case 2:
            CODA_CALL_MICROKERNEL(2);
            break;
        case 3:
            CODA_CALL_MICROKERNEL(3);
            break;
        case 4:
            if constexpr (col_vectors == 2) {
                if (accumulate) {
                    gemm_microkernel_4x2<true>(
                            a_base, b_panel, acc_base, K, k_begin, k_end);
                } else {
                    gemm_microkernel_4x2<false>(
                            a_base, b_panel, acc_base, K, k_begin, k_end);
                }
            } else if constexpr (col_vectors == 3) {
                if (accumulate) {
                    gemm_microkernel_4x3<true>(
                            a_base, b_panel, acc_base, K, k_begin, k_end);
                } else {
                    gemm_microkernel_4x3<false>(
                            a_base, b_panel, acc_base, K, k_begin, k_end);
                }
            } else {
                CODA_CALL_MICROKERNEL(4);
            }
            break;
        default:
            TORCH_CHECK(false, "unsupported ATen vector GEMM microkernel row count");
    }

#undef CODA_CALL_MICROKERNEL
}

template <int64_t col_vectors, typename Visitor>
void visit_gemm_vectors_blocked(
        const at::Tensor &A,
        const at::Tensor &B,
        const Visitor &visitor,
        int64_t requested_group_blocks = 0) {
    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);
    const float *a_base = A.data_ptr<float>();
    constexpr int64_t vec_size = Vec::size();
    constexpr int64_t row_tile = 4;
    constexpr int64_t block_n = col_vectors * vec_size;
    const int64_t num_blocks = ceil_div(N, block_n);
    const int64_t target_group_blocks = ceil_div(num_blocks, at::get_num_threads());
    const int64_t group_blocks = requested_group_blocks > 0
            ? requested_group_blocks
            : std::max<int64_t>(1, target_group_blocks);
    const int64_t num_groups = ceil_div(num_blocks, group_blocks);
    const int64_t mc_rows = std::max<int64_t>(
            row_tile,
            std::min<int64_t>(
                    kMaxMcRows,
                    (kL2Budget / (K * static_cast<int64_t>(sizeof(float))) / row_tile) *
                            row_tile));
    const int64_t raw_kc = kL1Budget / (block_n * static_cast<int64_t>(sizeof(float)));
    const int64_t kc = std::min<int64_t>(
            K,
            std::max<int64_t>(64, (raw_kc / 64) * 64));
    const auto packed_B = pack_b_register_blocks<col_vectors>(B);
    const float *packed_base = packed_B.template data_ptr<float>();

    at::parallel_for(0, num_groups, 1, [&](int64_t begin, int64_t end) {
        alignas(64) std::array<float, kMaxAccBufferSize> acc_buffer;
        for (int64_t group = begin; group < end; ++group) {
            const int64_t block_begin = group * group_blocks;
            const int64_t block_end = std::min<int64_t>(num_blocks, block_begin + group_blocks);
            for (int64_t m0 = 0; m0 < M; m0 += mc_rows) {
                const int64_t rows = std::min<int64_t>(mc_rows, M - m0);
                for (int64_t nb = block_begin; nb < block_end; ++nb) {
                    const float *b_panel = packed_base + nb * K * block_n;
                    for (int64_t k0 = 0; k0 < K; k0 += kc) {
                        const int64_t k_end = std::min<int64_t>(K, k0 + kc);
                        for (int64_t r0 = 0; r0 < rows; r0 += row_tile) {
                            const int64_t micro_rows = std::min<int64_t>(row_tile, rows - r0);
                            gemm_microkernel_rows<col_vectors>(
                                    micro_rows,
                                    k0 != 0,
                                    a_base + (m0 + r0) * K,
                                    b_panel,
                                    acc_buffer.data() + r0 * block_n,
                                    K,
                                    block_n,
                                    k0,
                                    k_end);
                        }
                    }
                    for (int64_t r = 0; r < rows; ++r) {
                        const int64_t n = nb * block_n;
                        const int64_t count = std::max<int64_t>(
                                0,
                                std::min<int64_t>(block_n, N - n));
                        if (count > 0) {
                            visitor(
                                    m0 + r,
                                    n,
                                    count,
                                    acc_buffer.data() + r * block_n);
                        }
                    }
                }
            }
        }
    });
}

template <typename Visitor>
void visit_gemm_vectors(
        const at::Tensor &A,
        const at::Tensor &B,
        const Visitor &visitor) {
    visit_gemm_vectors_blocked<3>(A, B, visitor);
}

template <typename Visitor>
void visit_gemm_vectors_reduction(
        const at::Tensor &A,
        const at::Tensor &B,
        int64_t reduction_block_size,
        const Visitor &visitor) {
    constexpr int64_t block_n = 3 * Vec::size();
    const int64_t num_blocks = ceil_div(B.size(1), block_n);
    const int64_t aligned_group_blocks =
            std::lcm(reduction_block_size, block_n) / block_n;
    const int64_t target_group_blocks = ceil_div(num_blocks, at::get_num_threads());
    const int64_t group_blocks = aligned_group_blocks *
            std::max<int64_t>(1, target_group_blocks / aligned_group_blocks);
    visit_gemm_vectors_blocked<3>(A, B, visitor, group_blocks);
}

at::Tensor execute_row_scale(
        const at::Tensor &A,
        const at::Tensor &B,
        const at::Tensor &R) {
    require_gemm_inputs(A, B);
    require_vector(R, A.size(0), "R");

    const int64_t M = A.size(0);
    const int64_t N = B.size(1);
    auto D = at::empty({M, N}, A.options().dtype(at::kFloat));
    const float *r_ptr = R.data_ptr<float>();
    float *d_ptr = D.data_ptr<float>();

    visit_gemm_vectors(A, B, [&](int64_t m, int64_t n, int64_t count, const float *values) {
        float *d_row = d_ptr + m * N;
        const Vec scale(r_ptr[m]);
        for (int64_t i = 0; i < count; i += Vec::size()) {
            const int64_t width = std::min<int64_t>(Vec::size(), count - i);
            const Vec value = Vec::loadu(values + i, width) * scale;
            value.store(d_row + n + i, width);
        }
    });

    return D;
}

std::pair<at::Tensor, at::Tensor> execute_swiglu(
        const at::Tensor &A,
        const at::Tensor &B,
        const at::Tensor *R) {
    require_gemm_inputs(A, B);
    if (R != nullptr) {
        require_vector(*R, A.size(0), "R");
    }
    TORCH_CHECK(B.size(1) % 2 == 0, "SwiGLU expects an even last dimension");

    const int64_t M = A.size(0);
    const int64_t N = B.size(1);
    auto D = at::empty({M, N}, A.options().dtype(at::kFloat));
    auto O = at::empty({M, N / 2}, A.options().dtype(at::kFloat));
    const float *r_ptr = R == nullptr ? nullptr : R->data_ptr<float>();
    float *d_ptr = D.data_ptr<float>();
    float *o_ptr = O.data_ptr<float>();

    visit_gemm_vectors(A, B, [&](int64_t m, int64_t n, int64_t count, const float *values) {
        float *d_row = d_ptr + m * N;
        float *o_row = o_ptr + m * (N / 2);
        const Vec scale(r_ptr == nullptr ? 1.0f : r_ptr[m]);
        int64_t i = 0;
        for (; i + 2 * Vec::size() <= count; i += 2 * Vec::size()) {
            const Vec value0 = Vec::loadu(values + i) * scale;
            const Vec value1 = Vec::loadu(values + i + Vec::size()) * scale;
            value0.store(d_row + n + i);
            value1.store(d_row + n + i + Vec::size());
            const auto gate_up = at::vec::deinterleave2(value0, value1);
            const Vec output =
                    gate_up.first / (Vec(1.0f) + (-gate_up.first).exp()) *
                    gate_up.second;
            output.store(o_row + (n + i) / 2);
        }
        if (i < count) {
            const int64_t width = count - i;
            const Vec value = Vec::loadu(values + i, width) * scale;
            value.store(d_row + n + i, width);
            const auto gate_up = at::vec::deinterleave2(value, Vec(0.0f));
            const Vec output =
                    gate_up.first / (Vec(1.0f) + (-gate_up.first).exp()) *
                    gate_up.second;
            output.store(o_row + (n + i) / 2, width / 2);
        }
    });

    return {D, O};
}

std::pair<at::Tensor, at::Tensor> execute_rope(
        const at::Tensor &A,
        const at::Tensor &B,
        const at::Tensor *R,
        const at::Tensor &cos_sin,
        bool backward) {
    require_gemm_inputs(A, B);
    if (R != nullptr) {
        require_vector(*R, A.size(0), "R");
    }
    require_matrix(cos_sin, "cos_sin");
    TORCH_CHECK(cos_sin.size(0) == A.size(0) && cos_sin.size(1) == B.size(1), "RoPE cos_sin shape must match accumulator shape");
    TORCH_CHECK(B.size(1) % 2 == 0, "RoPE expects an even last dimension");

    const int64_t M = A.size(0);
    const int64_t N = B.size(1);
    auto D = at::empty({M, N}, A.options().dtype(at::kFloat));
    auto O = at::empty({M, N}, A.options().dtype(at::kFloat));
    const float *r_ptr = R == nullptr ? nullptr : R->data_ptr<float>();
    const float *cs_ptr = cos_sin.data_ptr<float>();
    float *d_ptr = D.data_ptr<float>();
    float *o_ptr = O.data_ptr<float>();
    const float sign = backward ? -1.0f : 1.0f;

    visit_gemm_vectors(A, B, [&](int64_t m, int64_t n, int64_t count, const float *values) {
        float *d_row = d_ptr + m * N;
        float *o_row = o_ptr + m * N;
        const float *cs_row = cs_ptr + m * N;
        const Vec scale(r_ptr == nullptr ? 1.0f : r_ptr[m]);
        const Vec sign_vec(sign);
        int64_t i = 0;
        for (; i + 2 * Vec::size() <= count; i += 2 * Vec::size()) {
            const Vec value0 = Vec::loadu(values + i) * scale;
            const Vec value1 = Vec::loadu(values + i + Vec::size()) * scale;
            value0.store(d_row + n + i);
            value1.store(d_row + n + i + Vec::size());
            const auto x = at::vec::deinterleave2(value0, value1);
            const auto cs = at::vec::deinterleave2(
                    Vec::loadu(cs_row + n + i),
                    Vec::loadu(cs_row + n + i + Vec::size()));
            const Vec s = cs.second * sign_vec;
            const Vec y0 = x.first * cs.first + x.second * s;
            const Vec y1 = x.first * (-s) + x.second * cs.first;
            const auto output = at::vec::interleave2(y0, y1);
            output.first.store(o_row + n + i);
            output.second.store(o_row + n + i + Vec::size());
        }
        if (i < count) {
            const int64_t width = count - i;
            const Vec value = Vec::loadu(values + i, width) * scale;
            value.store(d_row + n + i, width);
            const auto x = at::vec::deinterleave2(value, Vec(0.0f));
            const auto cs = at::vec::deinterleave2(
                    Vec::loadu(cs_row + n + i, width),
                    Vec(0.0f));
            const Vec s = cs.second * sign_vec;
            const Vec y0 = x.first * cs.first + x.second * s;
            const Vec y1 = x.first * (-s) + x.second * cs.first;
            at::vec::interleave2(y0, y1).first.store(o_row + n + i, width);
        }
    });

    return {D, O};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> execute_residual_partial_rmsnorm(
        const at::Tensor &A,
        const at::Tensor &B,
        const at::Tensor &C,
        const at::Tensor &W,
        int64_t block_size) {
    require_gemm_inputs(A, B);
    require_matrix(C, "C");
    require_vector(W, B.size(1), "W");
    TORCH_CHECK(C.size(0) == A.size(0) && C.size(1) == B.size(1), "residual shape mismatch");
    TORCH_CHECK(block_size > 0, "block post-op requires a positive block_size");
    TORCH_CHECK(B.size(1) % block_size == 0, "N must be divisible by block_size");

    const int64_t M = A.size(0);
    const int64_t N = B.size(1);
    const int64_t num_blocks = N / block_size;
    auto D = at::empty({M, N}, A.options().dtype(at::kFloat));
    auto S = at::empty({M, num_blocks}, A.options().dtype(at::kFloat));
    auto O = at::empty({M, N}, A.options().dtype(at::kFloat));
    const float *c_base = C.data_ptr<float>();
    const float *w_ptr = W.data_ptr<float>();
    float *d_ptr = D.data_ptr<float>();
    float *s_ptr = S.data_ptr<float>();
    float *o_ptr = O.data_ptr<float>();
    S.zero_();

    visit_gemm_vectors_reduction(A, B, block_size, [&](int64_t m, int64_t n, int64_t count, const float *values) {
        float *d_row = d_ptr + m * N;
        float *s_row = s_ptr + m * num_blocks;
        float *o_row = o_ptr + m * N;
        const float *c_row = c_base + m * N;
        for (int64_t i = 0; i < count; i += Vec::size()) {
            const int64_t width = std::min<int64_t>(Vec::size(), count - i);
            const int64_t col = n + i;
            const Vec value =
                    Vec::loadu(values + i, width) + Vec::loadu(c_row + col, width);
            value.store(d_row + col, width);
            (value * Vec::loadu(w_ptr + col, width)).store(o_row + col, width);
            if (col / block_size == (col + width - 1) / block_size) {
                const Vec square = value * value;
                const float sum_sq = at::vec::vec_reduce_all<float>(
                        [](Vec &x, Vec &y) { return x + y; },
                        square,
                        width);
                s_row[col / block_size] +=
                        sum_sq / static_cast<float>(block_size);
            } else {
                alignas(64) std::array<float, Vec::size()> scalar_values;
                value.store(scalar_values.data(), width);
                for (int64_t j = 0; j < width; ++j) {
                    const float scalar = scalar_values[j];
                    s_row[(col + j) / block_size] +=
                            (scalar * scalar) / static_cast<float>(block_size);
                }
            }
        }
    });

    return {D, S, O};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> execute_partial_cross_entropy(
        const at::Tensor &A,
        const at::Tensor &B,
        const at::Tensor *R,
        const at::Tensor &targets,
        int64_t block_size) {
    require_gemm_inputs(A, B);
    if (R != nullptr) {
        require_vector(*R, A.size(0), "R");
    }
    require_targets(targets, A.size(0), "targets");
    TORCH_CHECK(block_size > 0, "block logsumexp requires a positive block_size");
    TORCH_CHECK(B.size(1) % block_size == 0, "N must be divisible by block_size");

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);
    const int64_t num_blocks = N / block_size;
    auto logits = at::empty({M, N}, A.options().dtype(at::kFloat));
    auto logits_tgt = at::empty({M}, A.options().dtype(at::kFloat));
    auto logits_lse = at::empty({M, num_blocks}, A.options().dtype(at::kFloat));
    const float *a_base = A.data_ptr<float>();
    const float *b_base = B.data_ptr<float>();
    const float *r_ptr = R == nullptr ? nullptr : R->data_ptr<float>();
    const int64_t *targets_ptr = targets.data_ptr<int64_t>();
    float *logits_ptr = logits.data_ptr<float>();
    float *logits_tgt_ptr = logits_tgt.data_ptr<float>();
    float *logits_lse_ptr = logits_lse.data_ptr<float>();
    constexpr int64_t vec_size = Vec::size();

    at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
        alignas(64) float values[vec_size];
        for (int64_t m = begin; m < end; ++m) {
            const int64_t target = targets_ptr[m];
            TORCH_CHECK(target >= 0 && target < N, "target index is out of bounds");
            bool target_seen = false;
            const float scale = r_ptr == nullptr ? 1.0f : r_ptr[m];
            for (int64_t block = 0; block < num_blocks; ++block) {
                float max_value = -std::numeric_limits<float>::infinity();
                float sum_exp = 0.0f;
                const int64_t block_start = block * block_size;
                const int64_t block_end = block_start + block_size;
                for (int64_t n = block_start; n < block_end; n += vec_size) {
                    const int64_t count = std::min<int64_t>(vec_size, block_end - n);
                    Vec acc(0.0f);
                    for (int64_t k = 0; k < K; ++k) {
                        const Vec a_vec(a_base[m * K + k]);
                        const Vec b_vec = Vec::loadu(b_base + k * N + n, count);
                        acc = at::vec::fmadd(a_vec, b_vec, acc);
                    }
                    acc.store(values, count);
                    float *logits_row = logits_ptr + m * N;
                    for (int64_t i = 0; i < count; ++i) {
                        const int64_t col = n + i;
                        const float value = values[i] * scale;
                        logits_row[col] = value;
                        if (col == target) {
                            logits_tgt_ptr[m] = value;
                            target_seen = true;
                        }
                        update_logsumexp(value, max_value, sum_exp);
                    }
                }
                logits_lse_ptr[m * num_blocks + block] = max_value + std::log(sum_exp);
            }
            TORCH_CHECK(target_seen, "target index was not visited");
        }
    });

    return {logits, logits_tgt, logits_lse};
}

bool is_residual_partial_rmsnorm(const std::vector<PostOpNode> &nodes) {
    return nodes.size() == 3 &&
            nodes[0].kind == PostOpKind::ResidualAdd &&
            nodes[1].kind == PostOpKind::BlockMeanSquareReduction &&
            nodes[2].kind == PostOpKind::RowVectorScaleSideOutput;
}

bool is_row_scale(const std::vector<PostOpNode> &nodes) {
    return nodes.size() == 1 && nodes[0].kind == PostOpKind::RowScale;
}

bool is_swiglu(const std::vector<PostOpNode> &nodes) {
    return nodes.size() == 1 && nodes[0].kind == PostOpKind::SwiGLU;
}

bool is_row_scale_swiglu(const std::vector<PostOpNode> &nodes) {
    return nodes.size() == 2 &&
            nodes[0].kind == PostOpKind::RowScale &&
            nodes[1].kind == PostOpKind::SwiGLU;
}

bool is_rope(const std::vector<PostOpNode> &nodes) {
    return nodes.size() == 1 && nodes[0].kind == PostOpKind::RoPE;
}

bool is_row_scale_rope(const std::vector<PostOpNode> &nodes) {
    return nodes.size() == 2 &&
            nodes[0].kind == PostOpKind::RowScale &&
            nodes[1].kind == PostOpKind::RoPE;
}

bool is_partial_cross_entropy(const std::vector<PostOpNode> &nodes) {
    return nodes.size() == 2 &&
            nodes[0].kind == PostOpKind::TargetLogitSelect &&
            nodes[1].kind == PostOpKind::BlockLogSumExp;
}

bool is_row_scale_partial_cross_entropy(const std::vector<PostOpNode> &nodes) {
    return nodes.size() == 3 &&
            nodes[0].kind == PostOpKind::RowScale &&
            nodes[1].kind == PostOpKind::TargetLogitSelect &&
            nodes[2].kind == PostOpKind::BlockLogSumExp;
}
#endif

}  // namespace

bool has_aten_vec() {
#if defined(CODA_CPU_WITH_ATEN_VEC)
    return true;
#else
    return false;
#endif
}

std::string aten_vec_isa() {
#if defined(CODA_CPU_ATEN_VEC_ISA_AVX2)
    return "avx2";
#elif defined(CODA_CPU_ATEN_VEC_ISA_GENERIC)
    return "generic";
#else
    return "unavailable";
#endif
}

std::pair<at::Tensor, py::dict> execute_aten_vec_postops(
        const std::string &program_name,
        const py::list &nodes,
        const at::Tensor &A,
        const at::Tensor &B,
        const py::dict &tensors) {
    (void)program_name;
#if !defined(CODA_CPU_WITH_ATEN_VEC)
    (void)nodes;
    (void)A;
    (void)B;
    (void)tensors;
    throw std::runtime_error("ATen vector provider was not compiled into this native extension");
#else
    const auto parsed_nodes = parse_post_ops(nodes);
    const auto tensor_map = parse_tensor_map(tensors);
    py::dict side_outputs;

    if (is_residual_partial_rmsnorm(parsed_nodes)) {
        const auto &C = require_tensor(tensor_map, require_name(parsed_nodes[0].tensor, "tensor"));
        const auto &W = require_tensor(tensor_map, require_name(parsed_nodes[2].tensor, "tensor"));
        auto outputs = execute_residual_partial_rmsnorm(A, B, C, W, parsed_nodes[1].block_size);
        side_outputs[py::str(require_name(parsed_nodes[1].output, "output"))] = std::get<1>(outputs);
        side_outputs[py::str(require_name(parsed_nodes[2].output, "output"))] = std::get<2>(outputs);
        return {std::get<0>(outputs), side_outputs};
    }

    if (is_row_scale(parsed_nodes)) {
        const auto &R = require_tensor(tensor_map, require_name(parsed_nodes[0].tensor, "tensor"));
        return {execute_row_scale(A, B, R), side_outputs};
    }

    if (is_swiglu(parsed_nodes)) {
        auto outputs = execute_swiglu(A, B, nullptr);
        side_outputs[py::str(require_name(parsed_nodes[0].output, "output"))] = outputs.second;
        return {outputs.first, side_outputs};
    }

    if (is_row_scale_swiglu(parsed_nodes)) {
        const auto &R = require_tensor(tensor_map, require_name(parsed_nodes[0].tensor, "tensor"));
        auto outputs = execute_swiglu(A, B, &R);
        side_outputs[py::str(require_name(parsed_nodes[1].output, "output"))] = outputs.second;
        return {outputs.first, side_outputs};
    }

    if (is_rope(parsed_nodes)) {
        const auto &cos_sin = require_tensor(tensor_map, require_name(parsed_nodes[0].tensor, "tensor"));
        auto outputs = execute_rope(A, B, nullptr, cos_sin, parsed_nodes[0].backward);
        side_outputs[py::str(require_name(parsed_nodes[0].output, "output"))] = outputs.second;
        return {outputs.first, side_outputs};
    }

    if (is_row_scale_rope(parsed_nodes)) {
        const auto &R = require_tensor(tensor_map, require_name(parsed_nodes[0].tensor, "tensor"));
        const auto &cos_sin = require_tensor(tensor_map, require_name(parsed_nodes[1].tensor, "tensor"));
        auto outputs = execute_rope(A, B, &R, cos_sin, parsed_nodes[1].backward);
        side_outputs[py::str(require_name(parsed_nodes[1].output, "output"))] = outputs.second;
        return {outputs.first, side_outputs};
    }

    if (is_partial_cross_entropy(parsed_nodes)) {
        const auto &targets = require_tensor(tensor_map, require_name(parsed_nodes[0].tensor, "tensor"));
        auto outputs = execute_partial_cross_entropy(A, B, nullptr, targets, parsed_nodes[1].block_size);
        side_outputs[py::str(require_name(parsed_nodes[0].output, "output"))] = std::get<1>(outputs);
        side_outputs[py::str(require_name(parsed_nodes[1].output, "output"))] = std::get<2>(outputs);
        return {std::get<0>(outputs), side_outputs};
    }

    if (is_row_scale_partial_cross_entropy(parsed_nodes)) {
        const auto &R = require_tensor(tensor_map, require_name(parsed_nodes[0].tensor, "tensor"));
        const auto &targets = require_tensor(tensor_map, require_name(parsed_nodes[1].tensor, "tensor"));
        auto outputs = execute_partial_cross_entropy(A, B, &R, targets, parsed_nodes[2].block_size);
        side_outputs[py::str(require_name(parsed_nodes[1].output, "output"))] = std::get<1>(outputs);
        side_outputs[py::str(require_name(parsed_nodes[2].output, "output"))] = std::get<2>(outputs);
        return {std::get<0>(outputs), side_outputs};
    }

    throw std::invalid_argument("unsupported ATen vector post-op program");
#endif
}

}  // namespace coda::cpu
