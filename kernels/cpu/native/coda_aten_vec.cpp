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
#include <ATen/cpu/vec/functional.h>
#if defined(CODA_CPU_ATEN_VEC_ISA_SVE2_BF16)
#include "coda_aten_vec_sve_bf16.h"
#endif
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
constexpr int64_t kMaxMcRows = 128;
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
            B.data_ptr(),
            B.size(0),
            B.size(1),
            B.stride(0),
            B.stride(1),
            block_n,
            B.is_inference() ? 0 : B.unsafeGetTensorImpl()->version_counter().current_version(),
    };
}

int64_t ceil_div(int64_t x, int64_t y) {
    return (x + y - 1) / y;
}

template <typename T>
inline Vec load_vec_as_float(const T *ptr, int64_t count = Vec::size()) {
    if constexpr (std::is_same_v<T, float>) {
        return Vec::loadu(ptr, count);
    } else {
        auto bvec = at::vec::Vectorized<c10::BFloat16>::loadu(ptr, count);
        return std::get<0>(at::vec::convert_to_float(bvec));
    }
}

template <typename T>
inline void store_float_as_vec(T *ptr, const Vec &fvec, int64_t count = Vec::size()) {
    if constexpr (std::is_same_v<T, float>) {
        fvec.store(ptr, count);
    } else {
        auto bvec = at::vec::convert_from_float<c10::BFloat16>(fvec, Vec(0.0f));
        bvec.store(ptr, count);
    }
}

template <typename T>
struct PackedBType {
    using type = float;
};

#if defined(__AVX512BF16__) || defined(CODA_CPU_ATEN_VEC_ISA_SVE2_BF16)
template <>
struct PackedBType<c10::BFloat16> {
    using type = c10::BFloat16;
};
#endif

template <typename T>
inline int32_t load_a_pair(const T *a_base, int64_t row_offset, int64_t p, int64_t K) {
    (void)K;
    if constexpr (std::is_same_v<T, c10::BFloat16>) {
        typedef int32_t __attribute__((__may_alias__)) aliased_int32_t;
        return *reinterpret_cast<const aliased_int32_t*>(a_base + row_offset + 2 * p);
    } else {
        return 0;
    }
}

#if defined(CODA_CPU_ATEN_VEC_ISA_AVX512)
inline float vec_reduce_sum(const Vec &v) {
    __m512 val = v.values;
    __m256 ymm = _mm256_add_ps(_mm512_castps512_ps256(val), _mm512_extractf32x8_ps(val, 1));
    __m128 xmm = _mm_add_ps(_mm256_castps256_ps128(ymm), _mm256_extractf128_ps(ymm, 1));
    __m128 xmm_hi = _mm_movehl_ps(xmm, xmm);
    xmm = _mm_add_ps(xmm, xmm_hi);
    __m128 xmm_shuf = _mm_shuffle_ps(xmm, xmm, _MM_SHUFFLE(1, 1, 1, 1));
    xmm = _mm_add_ss(xmm, xmm_shuf);
    return _mm_cvtss_f32(xmm);
}
#else
inline float vec_reduce_sum(const Vec &v) {
    alignas(64) std::array<float, Vec::size()> scalar_values;
    v.store(scalar_values.data());
    float sum = 0.0f;
    for (int64_t i = 0; i < Vec::size(); ++i) {
        sum += scalar_values[i];
    }
    return sum;
}
#endif

inline void atomic_add(float* address, float val) {
    #if defined(__x86_64__) || defined(_M_X64)
    union {
        float f;
        uint32_t i;
    } old_val, new_val;
    do {
        old_val.f = *address;
        new_val.f = old_val.f + val;
    } while (!__sync_bool_compare_and_swap(reinterpret_cast<volatile uint32_t*>(address), old_val.i, new_val.i));
    #else
    #pragma omp atomic
    *address += val;
    #endif
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

template <typename T>
void require_matrix_template(const at::Tensor &x, const char *name) {
    TORCH_CHECK(x.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(x.dim() == 2, name, " must be a 2D tensor");
    if constexpr (std::is_same_v<T, float>) {
        TORCH_CHECK(x.scalar_type() == at::kFloat, name, " must be float32");
    } else {
        TORCH_CHECK(x.scalar_type() == at::kBFloat16, name, " must be bfloat16");
    }
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

template <typename T>
void require_vector_template(const at::Tensor &x, int64_t size, const char *name) {
    TORCH_CHECK(x.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(x.dim() == 1, name, " must be a 1D tensor");
    TORCH_CHECK(x.size(0) == size, name, " has an unexpected shape");
    if constexpr (std::is_same_v<T, float>) {
        TORCH_CHECK(x.scalar_type() == at::kFloat, name, " must be float32");
    } else {
        TORCH_CHECK(x.scalar_type() == at::kBFloat16, name, " must be bfloat16");
    }
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

template <typename T>
void require_gemm_inputs_template(const at::Tensor &A, const at::Tensor &B) {
    require_matrix_template<T>(A, "A");
    require_matrix_template<T>(B, "B");
    TORCH_CHECK(A.size(1) == B.size(0), "incompatible GEMM shapes");
    TORCH_CHECK(A.size(1) > 0, "GEMM K dimension must be positive");
    if constexpr (std::is_same_v<T, c10::BFloat16>) {
        TORCH_CHECK(A.size(1) % 2 == 0, "BFloat16 GEMM K dimension must be even");
    }
}

void update_logsumexp(float value, float &max_value, float &sum_exp) {
    if (value > max_value) {
        sum_exp = sum_exp * std::exp(max_value - value) + 1.0f;
        max_value = value;
    } else {
        sum_exp += std::exp(value - max_value);
    }
}

template <typename T, int64_t col_vectors>
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

    using PackedT = typename PackedBType<T>::type;

    at::Tensor packed;
    if constexpr (std::is_same_v<PackedT, c10::BFloat16>) {
        const int64_t K_padded = ceil_div(K, 2) * 2;
        packed = at::empty({num_blocks, K_padded / 2, 2 * block_n}, B.options().dtype(at::kBFloat16));
        const T *b_base = B.data_ptr<T>();
        PackedT *packed_base = packed.data_ptr<PackedT>();

        at::parallel_for(0, num_blocks, 1, [&](int64_t begin, int64_t end) {
            for (int64_t nb = begin; nb < end; ++nb) {
                for (int64_t p = 0; p < K_padded / 2; ++p) {
                    const int64_t k0 = 2 * p;
                    const int64_t k1 = 2 * p + 1;
                    PackedT *packed_row = packed_base + (nb * (K_padded / 2) + p) * (2 * block_n);
                    for (int64_t c = 0; c < col_vectors; ++c) {
                        const int64_t n = nb * block_n + c * vec_size;
                        const int64_t count = std::max<int64_t>(
                                0,
                                std::min<int64_t>(vec_size, N - n));
                        if (count == 0) {
                            std::fill_n(packed_row + c * 2 * vec_size, 2 * vec_size, static_cast<PackedT>(0.0f));
                        } else {
                            for (int64_t i = 0; i < count; ++i) {
                                packed_row[c * 2 * vec_size + 2 * i] = static_cast<PackedT>(b_base[k0 * N + n + i]);
                                if (k1 < K) {
                                    packed_row[c * 2 * vec_size + 2 * i + 1] = static_cast<PackedT>(b_base[k1 * N + n + i]);
                                } else {
                                    packed_row[c * 2 * vec_size + 2 * i + 1] = static_cast<PackedT>(0.0f);
                                }
                            }
                            if (count < vec_size) {
                                std::fill_n(packed_row + c * 2 * vec_size + 2 * count, 2 * (vec_size - count), static_cast<PackedT>(0.0f));
                            }
                        }
                    }
                }
            }
        });
    } else {
        packed = at::empty({num_blocks, K, block_n}, B.options().dtype(at::kFloat));
        const T *b_base = B.data_ptr<T>();
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
                        if (count == 0) {
                            std::fill_n(packed_row + c * vec_size, vec_size, 0.0f);
                        } else {
                            for (int64_t i = 0; i < count; ++i) {
                                packed_row[c * vec_size + i] = static_cast<float>(b_base[k * N + n + i]);
                            }
                            if (count < vec_size) {
                                std::fill_n(packed_row + c * vec_size + count, vec_size - count, 0.0f);
                            }
                        }
                    }
                }
            }
        });
    }

    {
        std::lock_guard<std::mutex> lock(packed_b_cache_mutex);
        if (packed_b_cache.size() >= 1024) {
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
inline Vec load_accumulator(const float *ptr, int64_t count = Vec::size()) {
    if constexpr (accumulate) {
        return Vec::loadu(ptr, count);
    } else {
        return Vec(0.0f);
    }
}

template <typename T, bool accumulate>
inline void gemm_microkernel_4x2(
        const T *a_base,
        const typename PackedBType<T>::type *b_panel,
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

    using PackedT = typename PackedBType<T>::type;
    if constexpr (std::is_same_v<PackedT, c10::BFloat16>) {
#if defined(__AVX512BF16__)
        const int64_t p_begin = k_begin / 2;
        const int64_t p_end = (k_end + 1) / 2;

        __m512 c00_v = c00.values;
        __m512 c01_v = c01.values;
        __m512 c10_v = c10.values;
        __m512 c11_v = c11.values;
        __m512 c20_v = c20.values;
        __m512 c21_v = c21.values;
        __m512 c30_v = c30.values;
        __m512 c31_v = c31.values;

        const PackedT *b_ptr = b_panel + p_begin * 2 * block_n;
        const T *a0_ptr = a_base + 2 * p_begin;
        const T *a1_ptr = a0_ptr + K;
        const T *a2_ptr = a0_ptr + 2 * K;
        const T *a3_ptr = a0_ptr + 3 * K;
        typedef int32_t __attribute__((__may_alias__)) aliased_int32_t;

        int64_t p = p_begin;
        for (; p + 1 < p_end; p += 2) {
            __m512 b0 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr));
            __m512 b1 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + vec_size * 2));

            __m512 a0 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a0_ptr)));
            c00_v = _mm512_dpbf16_ps(c00_v, (__m512bh)a0, (__m512bh)b0);
            c01_v = _mm512_dpbf16_ps(c01_v, (__m512bh)a0, (__m512bh)b1);

            __m512 a1 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a1_ptr)));
            c10_v = _mm512_dpbf16_ps(c10_v, (__m512bh)a1, (__m512bh)b0);
            c11_v = _mm512_dpbf16_ps(c11_v, (__m512bh)a1, (__m512bh)b1);

            __m512 a2 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a2_ptr)));
            c20_v = _mm512_dpbf16_ps(c20_v, (__m512bh)a2, (__m512bh)b0);
            c21_v = _mm512_dpbf16_ps(c21_v, (__m512bh)a2, (__m512bh)b1);

            __m512 a3 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a3_ptr)));
            c30_v = _mm512_dpbf16_ps(c30_v, (__m512bh)a3, (__m512bh)b0);
            c31_v = _mm512_dpbf16_ps(c31_v, (__m512bh)a3, (__m512bh)b1);

            b0 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + 2 * block_n));
            b1 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + 2 * block_n + vec_size * 2));

            a0 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a0_ptr + 2)));
            c00_v = _mm512_dpbf16_ps(c00_v, (__m512bh)a0, (__m512bh)b0);
            c01_v = _mm512_dpbf16_ps(c01_v, (__m512bh)a0, (__m512bh)b1);

            a1 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a1_ptr + 2)));
            c10_v = _mm512_dpbf16_ps(c10_v, (__m512bh)a1, (__m512bh)b0);
            c11_v = _mm512_dpbf16_ps(c11_v, (__m512bh)a1, (__m512bh)b1);

            a2 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a2_ptr + 2)));
            c20_v = _mm512_dpbf16_ps(c20_v, (__m512bh)a2, (__m512bh)b0);
            c21_v = _mm512_dpbf16_ps(c21_v, (__m512bh)a2, (__m512bh)b1);

            a3 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a3_ptr + 2)));
            c30_v = _mm512_dpbf16_ps(c30_v, (__m512bh)a3, (__m512bh)b0);
            c31_v = _mm512_dpbf16_ps(c31_v, (__m512bh)a3, (__m512bh)b1);

            b_ptr += 4 * block_n;
            a0_ptr += 4;
            a1_ptr += 4;
            a2_ptr += 4;
            a3_ptr += 4;
        }
        for (; p < p_end; ++p) {
            __m512 b0 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr));
            __m512 b1 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + vec_size * 2));

            __m512 a0 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a0_ptr)));
            c00_v = _mm512_dpbf16_ps(c00_v, (__m512bh)a0, (__m512bh)b0);
            c01_v = _mm512_dpbf16_ps(c01_v, (__m512bh)a0, (__m512bh)b1);

            __m512 a1 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a1_ptr)));
            c10_v = _mm512_dpbf16_ps(c10_v, (__m512bh)a1, (__m512bh)b0);
            c11_v = _mm512_dpbf16_ps(c11_v, (__m512bh)a1, (__m512bh)b1);

            __m512 a2 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a2_ptr)));
            c20_v = _mm512_dpbf16_ps(c20_v, (__m512bh)a2, (__m512bh)b0);
            c21_v = _mm512_dpbf16_ps(c21_v, (__m512bh)a2, (__m512bh)b1);

            __m512 a3 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a3_ptr)));
            c30_v = _mm512_dpbf16_ps(c30_v, (__m512bh)a3, (__m512bh)b0);
            c31_v = _mm512_dpbf16_ps(c31_v, (__m512bh)a3, (__m512bh)b1);

            b_ptr += 2 * block_n;
            a0_ptr += 2;
            a1_ptr += 2;
            a2_ptr += 2;
            a3_ptr += 2;
        }

        c00 = Vec(c00_v);
        c01 = Vec(c01_v);
        c10 = Vec(c10_v);
        c11 = Vec(c11_v);
        c20 = Vec(c20_v);
        c21 = Vec(c21_v);
        c30 = Vec(c30_v);
        c31 = Vec(c31_v);
#elif defined(CODA_CPU_ATEN_VEC_ISA_SVE2_BF16)
        aten_vec_sve_bf16::gemm_microkernel<T, 4, 2, accumulate>(
                a_base, b_panel, acc_base, K, block_n, k_begin, k_end);
        return;
#else
        TORCH_CHECK(false, "BFloat16 packed GEMM requires AVX512_BF16 or AArch64 SVE BF16 support");
#endif
    } else {
        int64_t k = k_begin;
        for (; k + 1 < k_end; k += 2) {
            const Vec b0_0 = Vec::loadu(b_panel + k * block_n);
            const Vec b1_0 = Vec::loadu(b_panel + k * block_n + vec_size);
            const Vec b0_1 = Vec::loadu(b_panel + (k + 1) * block_n);
            const Vec b1_1 = Vec::loadu(b_panel + (k + 1) * block_n + vec_size);

            Vec a0(static_cast<float>(a_base[k]));
            c00 = at::vec::fmadd(a0, b0_0, c00);
            c01 = at::vec::fmadd(a0, b1_0, c01);
            Vec a1(static_cast<float>(a_base[k + 1]));
            c00 = at::vec::fmadd(a1, b0_1, c00);
            c01 = at::vec::fmadd(a1, b1_1, c01);

            a0 = Vec(static_cast<float>(a_base[K + k]));
            c10 = at::vec::fmadd(a0, b0_0, c10);
            c11 = at::vec::fmadd(a0, b1_0, c11);
            a1 = Vec(static_cast<float>(a_base[K + k + 1]));
            c10 = at::vec::fmadd(a1, b0_1, c10);
            c11 = at::vec::fmadd(a1, b1_1, c11);

            a0 = Vec(static_cast<float>(a_base[2 * K + k]));
            c20 = at::vec::fmadd(a0, b0_0, c20);
            c21 = at::vec::fmadd(a0, b1_0, c21);
            a1 = Vec(static_cast<float>(a_base[2 * K + k + 1]));
            c20 = at::vec::fmadd(a1, b0_1, c20);
            c21 = at::vec::fmadd(a1, b1_1, c21);

            a0 = Vec(static_cast<float>(a_base[3 * K + k]));
            c30 = at::vec::fmadd(a0, b0_0, c30);
            c31 = at::vec::fmadd(a0, b1_0, c31);
            a1 = Vec(static_cast<float>(a_base[3 * K + k + 1]));
            c30 = at::vec::fmadd(a1, b0_1, c30);
            c31 = at::vec::fmadd(a1, b1_1, c31);
        }
        for (; k < k_end; ++k) {
            const Vec b0 = Vec::loadu(b_panel + k * block_n);
            const Vec b1 = Vec::loadu(b_panel + k * block_n + vec_size);
            Vec a(static_cast<float>(a_base[k]));
            c00 = at::vec::fmadd(a, b0, c00);
            c01 = at::vec::fmadd(a, b1, c01);
            a = Vec(static_cast<float>(a_base[K + k]));
            c10 = at::vec::fmadd(a, b0, c10);
            c11 = at::vec::fmadd(a, b1, c11);
            a = Vec(static_cast<float>(a_base[2 * K + k]));
            c20 = at::vec::fmadd(a, b0, c20);
            c21 = at::vec::fmadd(a, b1, c21);
            a = Vec(static_cast<float>(a_base[3 * K + k]));
            c30 = at::vec::fmadd(a, b0, c30);
            c31 = at::vec::fmadd(a, b1, c31);
        }
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

template <typename T, bool accumulate>
inline void gemm_microkernel_4x3(
        const T *a_base,
        const typename PackedBType<T>::type *b_panel,
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

    using PackedT = typename PackedBType<T>::type;
    if constexpr (std::is_same_v<PackedT, c10::BFloat16>) {
#if defined(__AVX512BF16__)
        const int64_t p_begin = k_begin / 2;
        const int64_t p_end = (k_end + 1) / 2;

        __m512 c00_v = c00.values;
        __m512 c01_v = c01.values;
        __m512 c02_v = c02.values;
        __m512 c10_v = c10.values;
        __m512 c11_v = c11.values;
        __m512 c12_v = c12.values;
        __m512 c20_v = c20.values;
        __m512 c21_v = c21.values;
        __m512 c22_v = c22.values;
        __m512 c30_v = c30.values;
        __m512 c31_v = c31.values;
        __m512 c32_v = c32.values;

        const PackedT *b_ptr = b_panel + p_begin * 2 * block_n;
        const T *a0_ptr = a_base + 2 * p_begin;
        const T *a1_ptr = a0_ptr + K;
        const T *a2_ptr = a0_ptr + 2 * K;
        const T *a3_ptr = a0_ptr + 3 * K;
        typedef int32_t __attribute__((__may_alias__)) aliased_int32_t;

        int64_t p = p_begin;
        for (; p + 1 < p_end; p += 2) {
            __m512 b0 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr));
            __m512 b1 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + vec_size * 2));
            __m512 b2 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + vec_size * 4));

            __m512 a0 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a0_ptr)));
            c00_v = _mm512_dpbf16_ps(c00_v, (__m512bh)a0, (__m512bh)b0);
            c01_v = _mm512_dpbf16_ps(c01_v, (__m512bh)a0, (__m512bh)b1);
            c02_v = _mm512_dpbf16_ps(c02_v, (__m512bh)a0, (__m512bh)b2);

            __m512 a1 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a1_ptr)));
            c10_v = _mm512_dpbf16_ps(c10_v, (__m512bh)a1, (__m512bh)b0);
            c11_v = _mm512_dpbf16_ps(c11_v, (__m512bh)a1, (__m512bh)b1);
            c12_v = _mm512_dpbf16_ps(c12_v, (__m512bh)a1, (__m512bh)b2);

            __m512 a2 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a2_ptr)));
            c20_v = _mm512_dpbf16_ps(c20_v, (__m512bh)a2, (__m512bh)b0);
            c21_v = _mm512_dpbf16_ps(c21_v, (__m512bh)a2, (__m512bh)b1);
            c22_v = _mm512_dpbf16_ps(c22_v, (__m512bh)a2, (__m512bh)b2);

            __m512 a3 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a3_ptr)));
            c30_v = _mm512_dpbf16_ps(c30_v, (__m512bh)a3, (__m512bh)b0);
            c31_v = _mm512_dpbf16_ps(c31_v, (__m512bh)a3, (__m512bh)b1);
            c32_v = _mm512_dpbf16_ps(c32_v, (__m512bh)a3, (__m512bh)b2);

            b0 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + 2 * block_n));
            b1 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + 2 * block_n + vec_size * 2));
            b2 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + 2 * block_n + vec_size * 4));

            a0 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a0_ptr + 2)));
            c00_v = _mm512_dpbf16_ps(c00_v, (__m512bh)a0, (__m512bh)b0);
            c01_v = _mm512_dpbf16_ps(c01_v, (__m512bh)a0, (__m512bh)b1);
            c02_v = _mm512_dpbf16_ps(c02_v, (__m512bh)a0, (__m512bh)b2);

            a1 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a1_ptr + 2)));
            c10_v = _mm512_dpbf16_ps(c10_v, (__m512bh)a1, (__m512bh)b0);
            c11_v = _mm512_dpbf16_ps(c11_v, (__m512bh)a1, (__m512bh)b1);
            c12_v = _mm512_dpbf16_ps(c12_v, (__m512bh)a1, (__m512bh)b2);

            a2 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a2_ptr + 2)));
            c20_v = _mm512_dpbf16_ps(c20_v, (__m512bh)a2, (__m512bh)b0);
            c21_v = _mm512_dpbf16_ps(c21_v, (__m512bh)a2, (__m512bh)b1);
            c22_v = _mm512_dpbf16_ps(c22_v, (__m512bh)a2, (__m512bh)b2);

            a3 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a3_ptr + 2)));
            c30_v = _mm512_dpbf16_ps(c30_v, (__m512bh)a3, (__m512bh)b0);
            c31_v = _mm512_dpbf16_ps(c31_v, (__m512bh)a3, (__m512bh)b1);
            c32_v = _mm512_dpbf16_ps(c32_v, (__m512bh)a3, (__m512bh)b2);

            b_ptr += 4 * block_n;
            a0_ptr += 4;
            a1_ptr += 4;
            a2_ptr += 4;
            a3_ptr += 4;
        }
        for (; p < p_end; ++p) {
            __m512 b0 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr));
            __m512 b1 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + vec_size * 2));
            __m512 b2 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + vec_size * 4));

            __m512 a0 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a0_ptr)));
            c00_v = _mm512_dpbf16_ps(c00_v, (__m512bh)a0, (__m512bh)b0);
            c01_v = _mm512_dpbf16_ps(c01_v, (__m512bh)a0, (__m512bh)b1);
            c02_v = _mm512_dpbf16_ps(c02_v, (__m512bh)a0, (__m512bh)b2);

            __m512 a1 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a1_ptr)));
            c10_v = _mm512_dpbf16_ps(c10_v, (__m512bh)a1, (__m512bh)b0);
            c11_v = _mm512_dpbf16_ps(c11_v, (__m512bh)a1, (__m512bh)b1);
            c12_v = _mm512_dpbf16_ps(c12_v, (__m512bh)a1, (__m512bh)b2);

            __m512 a2 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a2_ptr)));
            c20_v = _mm512_dpbf16_ps(c20_v, (__m512bh)a2, (__m512bh)b0);
            c21_v = _mm512_dpbf16_ps(c21_v, (__m512bh)a2, (__m512bh)b1);
            c22_v = _mm512_dpbf16_ps(c22_v, (__m512bh)a2, (__m512bh)b2);

            __m512 a3 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a3_ptr)));
            c30_v = _mm512_dpbf16_ps(c30_v, (__m512bh)a3, (__m512bh)b0);
            c31_v = _mm512_dpbf16_ps(c31_v, (__m512bh)a3, (__m512bh)b1);
            c32_v = _mm512_dpbf16_ps(c32_v, (__m512bh)a3, (__m512bh)b2);

            b_ptr += 2 * block_n;
            a0_ptr += 2;
            a1_ptr += 2;
            a2_ptr += 2;
            a3_ptr += 2;
        }

        c00 = Vec(c00_v);
        c01 = Vec(c01_v);
        c02 = Vec(c02_v);
        c10 = Vec(c10_v);
        c11 = Vec(c11_v);
        c12 = Vec(c12_v);
        c20 = Vec(c20_v);
        c21 = Vec(c21_v);
        c22 = Vec(c22_v);
        c30 = Vec(c30_v);
        c31 = Vec(c31_v);
        c32 = Vec(c32_v);
#elif defined(CODA_CPU_ATEN_VEC_ISA_SVE2_BF16)
        aten_vec_sve_bf16::gemm_microkernel<T, 4, 3, accumulate>(
                a_base, b_panel, acc_base, K, block_n, k_begin, k_end);
        return;
#else
        TORCH_CHECK(false, "BFloat16 packed GEMM requires AVX512_BF16 or AArch64 SVE BF16 support");
#endif
    } else {
        int64_t k = k_begin;
        for (; k + 1 < k_end; k += 2) {
            const Vec b0_0 = Vec::loadu(b_panel + k * block_n);
            const Vec b1_0 = Vec::loadu(b_panel + k * block_n + vec_size);
            const Vec b2_0 = Vec::loadu(b_panel + k * block_n + 2 * vec_size);
            const Vec b0_1 = Vec::loadu(b_panel + (k + 1) * block_n);
            const Vec b1_1 = Vec::loadu(b_panel + (k + 1) * block_n + vec_size);
            const Vec b2_1 = Vec::loadu(b_panel + (k + 1) * block_n + 2 * vec_size);

            Vec a0(static_cast<float>(a_base[k]));
            c00 = at::vec::fmadd(a0, b0_0, c00);
            c01 = at::vec::fmadd(a0, b1_0, c01);
            c02 = at::vec::fmadd(a0, b2_0, c02);
            Vec a1(static_cast<float>(a_base[k + 1]));
            c00 = at::vec::fmadd(a1, b0_1, c00);
            c01 = at::vec::fmadd(a1, b1_1, c01);
            c02 = at::vec::fmadd(a1, b2_1, c02);

            a0 = Vec(static_cast<float>(a_base[K + k]));
            c10 = at::vec::fmadd(a0, b0_0, c10);
            c11 = at::vec::fmadd(a0, b1_0, c11);
            c12 = at::vec::fmadd(a0, b2_0, c12);
            a1 = Vec(static_cast<float>(a_base[K + k + 1]));
            c10 = at::vec::fmadd(a1, b0_1, c10);
            c11 = at::vec::fmadd(a1, b1_1, c11);
            c12 = at::vec::fmadd(a1, b2_1, c12);

            a0 = Vec(static_cast<float>(a_base[2 * K + k]));
            c20 = at::vec::fmadd(a0, b0_0, c20);
            c21 = at::vec::fmadd(a0, b1_0, c21);
            c22 = at::vec::fmadd(a0, b2_0, c22);
            a1 = Vec(static_cast<float>(a_base[2 * K + k + 1]));
            c20 = at::vec::fmadd(a1, b0_1, c20);
            c21 = at::vec::fmadd(a1, b1_1, c21);
            c22 = at::vec::fmadd(a1, b2_1, c22);

            a0 = Vec(static_cast<float>(a_base[3 * K + k]));
            c30 = at::vec::fmadd(a0, b0_0, c30);
            c31 = at::vec::fmadd(a0, b1_0, c31);
            c32 = at::vec::fmadd(a0, b2_0, c32);
            a1 = Vec(static_cast<float>(a_base[3 * K + k + 1]));
            c30 = at::vec::fmadd(a1, b0_1, c30);
            c31 = at::vec::fmadd(a1, b1_1, c31);
            c32 = at::vec::fmadd(a1, b2_1, c32);
        }
        for (; k < k_end; ++k) {
            const Vec b0 = Vec::loadu(b_panel + k * block_n);
            const Vec b1 = Vec::loadu(b_panel + k * block_n + vec_size);
            const Vec b2 = Vec::loadu(b_panel + k * block_n + 2 * vec_size);
            Vec a(static_cast<float>(a_base[k]));
            c00 = at::vec::fmadd(a, b0, c00);
            c01 = at::vec::fmadd(a, b1, c01);
            c02 = at::vec::fmadd(a, b2, c02);
            a = Vec(static_cast<float>(a_base[K + k]));
            c10 = at::vec::fmadd(a, b0, c10);
            c11 = at::vec::fmadd(a, b1, c11);
            c12 = at::vec::fmadd(a, b2, c12);
            a = Vec(static_cast<float>(a_base[2 * K + k]));
            c20 = at::vec::fmadd(a, b0, c20);
            c21 = at::vec::fmadd(a, b1, c21);
            c22 = at::vec::fmadd(a, b2, c22);
            a = Vec(static_cast<float>(a_base[3 * K + k]));
            c30 = at::vec::fmadd(a, b0, c30);
            c31 = at::vec::fmadd(a, b1, c31);
            c32 = at::vec::fmadd(a, b2, c32);
        }
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

template <typename T, bool accumulate>
inline void gemm_microkernel_4x4(
        const T *a_base,
        const typename PackedBType<T>::type *b_panel,
        float *acc_base,
        int64_t K,
        int64_t k_begin,
        int64_t k_end) {
    constexpr int64_t vec_size = Vec::size();
    constexpr int64_t block_n = 4 * vec_size;
    Vec c00 = load_accumulator<accumulate>(acc_base);
    Vec c01 = load_accumulator<accumulate>(acc_base + vec_size);
    Vec c02 = load_accumulator<accumulate>(acc_base + 2 * vec_size);
    Vec c03 = load_accumulator<accumulate>(acc_base + 3 * vec_size);
    Vec c10 = load_accumulator<accumulate>(acc_base + block_n);
    Vec c11 = load_accumulator<accumulate>(acc_base + block_n + vec_size);
    Vec c12 = load_accumulator<accumulate>(acc_base + block_n + 2 * vec_size);
    Vec c13 = load_accumulator<accumulate>(acc_base + block_n + 3 * vec_size);
    Vec c20 = load_accumulator<accumulate>(acc_base + 2 * block_n);
    Vec c21 = load_accumulator<accumulate>(acc_base + 2 * block_n + vec_size);
    Vec c22 = load_accumulator<accumulate>(acc_base + 2 * block_n + 2 * vec_size);
    Vec c23 = load_accumulator<accumulate>(acc_base + 2 * block_n + 3 * vec_size);
    Vec c30 = load_accumulator<accumulate>(acc_base + 3 * block_n);
    Vec c31 = load_accumulator<accumulate>(acc_base + 3 * block_n + vec_size);
    Vec c32 = load_accumulator<accumulate>(acc_base + 3 * block_n + 2 * vec_size);
    Vec c33 = load_accumulator<accumulate>(acc_base + 3 * block_n + 3 * vec_size);

    using PackedT = typename PackedBType<T>::type;
    if constexpr (std::is_same_v<PackedT, c10::BFloat16>) {
#if defined(__AVX512BF16__)
        const int64_t p_begin = k_begin / 2;
        const int64_t p_end = (k_end + 1) / 2;

        __m512 c00_v = c00.values;
        __m512 c01_v = c01.values;
        __m512 c02_v = c02.values;
        __m512 c03_v = c03.values;
        __m512 c10_v = c10.values;
        __m512 c11_v = c11.values;
        __m512 c12_v = c12.values;
        __m512 c13_v = c13.values;
        __m512 c20_v = c20.values;
        __m512 c21_v = c21.values;
        __m512 c22_v = c22.values;
        __m512 c23_v = c23.values;
        __m512 c30_v = c30.values;
        __m512 c31_v = c31.values;
        __m512 c32_v = c32.values;
        __m512 c33_v = c33.values;

        const PackedT *b_ptr = b_panel + p_begin * 2 * block_n;
        const T *a0_ptr = a_base + 2 * p_begin;
        const T *a1_ptr = a0_ptr + K;
        const T *a2_ptr = a0_ptr + 2 * K;
        const T *a3_ptr = a0_ptr + 3 * K;
        typedef int32_t __attribute__((__may_alias__)) aliased_int32_t;

        int64_t p = p_begin;
        for (; p + 3 < p_end; p += 4) {
            __m512 b0 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr));
            __m512 b1 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + vec_size * 2));
            __m512 b2 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + vec_size * 4));
            __m512 b3 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + vec_size * 6));

            __m512 a0 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a0_ptr)));
            c00_v = _mm512_dpbf16_ps(c00_v, (__m512bh)a0, (__m512bh)b0);
            c01_v = _mm512_dpbf16_ps(c01_v, (__m512bh)a0, (__m512bh)b1);
            c02_v = _mm512_dpbf16_ps(c02_v, (__m512bh)a0, (__m512bh)b2);
            c03_v = _mm512_dpbf16_ps(c03_v, (__m512bh)a0, (__m512bh)b3);

            __m512 a1 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a1_ptr)));
            c10_v = _mm512_dpbf16_ps(c10_v, (__m512bh)a1, (__m512bh)b0);
            c11_v = _mm512_dpbf16_ps(c11_v, (__m512bh)a1, (__m512bh)b1);
            c12_v = _mm512_dpbf16_ps(c12_v, (__m512bh)a1, (__m512bh)b2);
            c13_v = _mm512_dpbf16_ps(c13_v, (__m512bh)a1, (__m512bh)b3);

            __m512 a2 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a2_ptr)));
            c20_v = _mm512_dpbf16_ps(c20_v, (__m512bh)a2, (__m512bh)b0);
            c21_v = _mm512_dpbf16_ps(c21_v, (__m512bh)a2, (__m512bh)b1);
            c22_v = _mm512_dpbf16_ps(c22_v, (__m512bh)a2, (__m512bh)b2);
            c23_v = _mm512_dpbf16_ps(c23_v, (__m512bh)a2, (__m512bh)b3);

            __m512 a3 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a3_ptr)));
            c30_v = _mm512_dpbf16_ps(c30_v, (__m512bh)a3, (__m512bh)b0);
            c31_v = _mm512_dpbf16_ps(c31_v, (__m512bh)a3, (__m512bh)b1);
            c32_v = _mm512_dpbf16_ps(c32_v, (__m512bh)a3, (__m512bh)b2);
            c33_v = _mm512_dpbf16_ps(c33_v, (__m512bh)a3, (__m512bh)b3);

            b0 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + 2 * block_n));
            b1 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + 2 * block_n + vec_size * 2));
            b2 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + 2 * block_n + vec_size * 4));
            b3 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + 2 * block_n + vec_size * 6));

            a0 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a0_ptr + 2)));
            c00_v = _mm512_dpbf16_ps(c00_v, (__m512bh)a0, (__m512bh)b0);
            c01_v = _mm512_dpbf16_ps(c01_v, (__m512bh)a0, (__m512bh)b1);
            c02_v = _mm512_dpbf16_ps(c02_v, (__m512bh)a0, (__m512bh)b2);
            c03_v = _mm512_dpbf16_ps(c03_v, (__m512bh)a0, (__m512bh)b3);

            a1 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a1_ptr + 2)));
            c10_v = _mm512_dpbf16_ps(c10_v, (__m512bh)a1, (__m512bh)b0);
            c11_v = _mm512_dpbf16_ps(c11_v, (__m512bh)a1, (__m512bh)b1);
            c12_v = _mm512_dpbf16_ps(c12_v, (__m512bh)a1, (__m512bh)b2);
            c13_v = _mm512_dpbf16_ps(c13_v, (__m512bh)a1, (__m512bh)b3);

            a2 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a2_ptr + 2)));
            c20_v = _mm512_dpbf16_ps(c20_v, (__m512bh)a2, (__m512bh)b0);
            c21_v = _mm512_dpbf16_ps(c21_v, (__m512bh)a2, (__m512bh)b1);
            c22_v = _mm512_dpbf16_ps(c22_v, (__m512bh)a2, (__m512bh)b2);
            c23_v = _mm512_dpbf16_ps(c23_v, (__m512bh)a2, (__m512bh)b3);

            a3 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a3_ptr + 2)));
            c30_v = _mm512_dpbf16_ps(c30_v, (__m512bh)a3, (__m512bh)b0);
            c31_v = _mm512_dpbf16_ps(c31_v, (__m512bh)a3, (__m512bh)b1);
            c32_v = _mm512_dpbf16_ps(c32_v, (__m512bh)a3, (__m512bh)b2);
            c33_v = _mm512_dpbf16_ps(c33_v, (__m512bh)a3, (__m512bh)b3);

            b0 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + 4 * block_n));
            b1 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + 4 * block_n + vec_size * 2));
            b2 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + 4 * block_n + vec_size * 4));
            b3 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + 4 * block_n + vec_size * 6));

            a0 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a0_ptr + 4)));
            c00_v = _mm512_dpbf16_ps(c00_v, (__m512bh)a0, (__m512bh)b0);
            c01_v = _mm512_dpbf16_ps(c01_v, (__m512bh)a0, (__m512bh)b1);
            c02_v = _mm512_dpbf16_ps(c02_v, (__m512bh)a0, (__m512bh)b2);
            c03_v = _mm512_dpbf16_ps(c03_v, (__m512bh)a0, (__m512bh)b3);

            a1 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a1_ptr + 4)));
            c10_v = _mm512_dpbf16_ps(c10_v, (__m512bh)a1, (__m512bh)b0);
            c11_v = _mm512_dpbf16_ps(c11_v, (__m512bh)a1, (__m512bh)b1);
            c12_v = _mm512_dpbf16_ps(c12_v, (__m512bh)a1, (__m512bh)b2);
            c13_v = _mm512_dpbf16_ps(c13_v, (__m512bh)a1, (__m512bh)b3);

            a2 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a2_ptr + 4)));
            c20_v = _mm512_dpbf16_ps(c20_v, (__m512bh)a2, (__m512bh)b0);
            c21_v = _mm512_dpbf16_ps(c21_v, (__m512bh)a2, (__m512bh)b1);
            c22_v = _mm512_dpbf16_ps(c22_v, (__m512bh)a2, (__m512bh)b2);
            c23_v = _mm512_dpbf16_ps(c23_v, (__m512bh)a2, (__m512bh)b3);

            a3 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a3_ptr + 4)));
            c30_v = _mm512_dpbf16_ps(c30_v, (__m512bh)a3, (__m512bh)b0);
            c31_v = _mm512_dpbf16_ps(c31_v, (__m512bh)a3, (__m512bh)b1);
            c32_v = _mm512_dpbf16_ps(c32_v, (__m512bh)a3, (__m512bh)b2);
            c33_v = _mm512_dpbf16_ps(c33_v, (__m512bh)a3, (__m512bh)b3);

            b0 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + 6 * block_n));
            b1 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + 6 * block_n + vec_size * 2));
            b2 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + 6 * block_n + vec_size * 4));
            b3 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + 6 * block_n + vec_size * 6));

            a0 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a0_ptr + 6)));
            c00_v = _mm512_dpbf16_ps(c00_v, (__m512bh)a0, (__m512bh)b0);
            c01_v = _mm512_dpbf16_ps(c01_v, (__m512bh)a0, (__m512bh)b1);
            c02_v = _mm512_dpbf16_ps(c02_v, (__m512bh)a0, (__m512bh)b2);
            c03_v = _mm512_dpbf16_ps(c03_v, (__m512bh)a0, (__m512bh)b3);

            a1 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a1_ptr + 6)));
            c10_v = _mm512_dpbf16_ps(c10_v, (__m512bh)a1, (__m512bh)b0);
            c11_v = _mm512_dpbf16_ps(c11_v, (__m512bh)a1, (__m512bh)b1);
            c12_v = _mm512_dpbf16_ps(c12_v, (__m512bh)a1, (__m512bh)b2);
            c13_v = _mm512_dpbf16_ps(c13_v, (__m512bh)a1, (__m512bh)b3);

            a2 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a2_ptr + 6)));
            c20_v = _mm512_dpbf16_ps(c20_v, (__m512bh)a2, (__m512bh)b0);
            c21_v = _mm512_dpbf16_ps(c21_v, (__m512bh)a2, (__m512bh)b1);
            c22_v = _mm512_dpbf16_ps(c22_v, (__m512bh)a2, (__m512bh)b2);
            c23_v = _mm512_dpbf16_ps(c23_v, (__m512bh)a2, (__m512bh)b3);

            a3 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a3_ptr + 6)));
            c30_v = _mm512_dpbf16_ps(c30_v, (__m512bh)a3, (__m512bh)b0);
            c31_v = _mm512_dpbf16_ps(c31_v, (__m512bh)a3, (__m512bh)b1);
            c32_v = _mm512_dpbf16_ps(c32_v, (__m512bh)a3, (__m512bh)b2);
            c33_v = _mm512_dpbf16_ps(c33_v, (__m512bh)a3, (__m512bh)b3);

            b_ptr += 8 * block_n;
            a0_ptr += 8;
            a1_ptr += 8;
            a2_ptr += 8;
            a3_ptr += 8;
        }
        for (; p < p_end; ++p) {
            __m512 b0 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr));
            __m512 b1 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + vec_size * 2));
            __m512 b2 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + vec_size * 4));
            __m512 b3 = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + vec_size * 6));

            __m512 a0 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a0_ptr)));
            c00_v = _mm512_dpbf16_ps(c00_v, (__m512bh)a0, (__m512bh)b0);
            c01_v = _mm512_dpbf16_ps(c01_v, (__m512bh)a0, (__m512bh)b1);
            c02_v = _mm512_dpbf16_ps(c02_v, (__m512bh)a0, (__m512bh)b2);
            c03_v = _mm512_dpbf16_ps(c03_v, (__m512bh)a0, (__m512bh)b3);

            __m512 a1 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a1_ptr)));
            c10_v = _mm512_dpbf16_ps(c10_v, (__m512bh)a1, (__m512bh)b0);
            c11_v = _mm512_dpbf16_ps(c11_v, (__m512bh)a1, (__m512bh)b1);
            c12_v = _mm512_dpbf16_ps(c12_v, (__m512bh)a1, (__m512bh)b2);
            c13_v = _mm512_dpbf16_ps(c13_v, (__m512bh)a1, (__m512bh)b3);

            __m512 a2 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a2_ptr)));
            c20_v = _mm512_dpbf16_ps(c20_v, (__m512bh)a2, (__m512bh)b0);
            c21_v = _mm512_dpbf16_ps(c21_v, (__m512bh)a2, (__m512bh)b1);
            c22_v = _mm512_dpbf16_ps(c22_v, (__m512bh)a2, (__m512bh)b2);
            c23_v = _mm512_dpbf16_ps(c23_v, (__m512bh)a2, (__m512bh)b3);

            __m512 a3 = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a3_ptr)));
            c30_v = _mm512_dpbf16_ps(c30_v, (__m512bh)a3, (__m512bh)b0);
            c31_v = _mm512_dpbf16_ps(c31_v, (__m512bh)a3, (__m512bh)b1);
            c32_v = _mm512_dpbf16_ps(c32_v, (__m512bh)a3, (__m512bh)b2);
            c33_v = _mm512_dpbf16_ps(c33_v, (__m512bh)a3, (__m512bh)b3);

            b_ptr += 2 * block_n;
            a0_ptr += 2;
            a1_ptr += 2;
            a2_ptr += 2;
            a3_ptr += 2;
        }

        c00 = Vec(c00_v);
        c01 = Vec(c01_v);
        c02 = Vec(c02_v);
        c03 = Vec(c03_v);
        c10 = Vec(c10_v);
        c11 = Vec(c11_v);
        c12 = Vec(c12_v);
        c13 = Vec(c13_v);
        c20 = Vec(c20_v);
        c21 = Vec(c21_v);
        c22 = Vec(c22_v);
        c23 = Vec(c23_v);
        c30 = Vec(c30_v);
        c31 = Vec(c31_v);
        c32 = Vec(c32_v);
        c33 = Vec(c33_v);
#elif defined(CODA_CPU_ATEN_VEC_ISA_SVE2_BF16)
        aten_vec_sve_bf16::gemm_microkernel<T, 4, 4, accumulate>(
                a_base, b_panel, acc_base, K, block_n, k_begin, k_end);
        return;
#else
        TORCH_CHECK(false, "BFloat16 packed GEMM requires AVX512_BF16 or AArch64 SVE BF16 support");
#endif
    } else {
        int64_t k = k_begin;
        for (; k + 1 < k_end; k += 2) {
            const Vec b0_0 = Vec::loadu(b_panel + k * block_n);
            const Vec b1_0 = Vec::loadu(b_panel + k * block_n + vec_size);
            const Vec b2_0 = Vec::loadu(b_panel + k * block_n + 2 * vec_size);
            const Vec b3_0 = Vec::loadu(b_panel + k * block_n + 3 * vec_size);
            const Vec b0_1 = Vec::loadu(b_panel + (k + 1) * block_n);
            const Vec b1_1 = Vec::loadu(b_panel + (k + 1) * block_n + vec_size);
            const Vec b2_1 = Vec::loadu(b_panel + (k + 1) * block_n + 2 * vec_size);
            const Vec b3_1 = Vec::loadu(b_panel + (k + 1) * block_n + 3 * vec_size);

            Vec a0(static_cast<float>(a_base[k]));
            c00 = at::vec::fmadd(a0, b0_0, c00);
            c01 = at::vec::fmadd(a0, b1_0, c01);
            c02 = at::vec::fmadd(a0, b2_0, c02);
            c03 = at::vec::fmadd(a0, b3_0, c03);
            Vec a1(static_cast<float>(a_base[k + 1]));
            c00 = at::vec::fmadd(a1, b0_1, c00);
            c01 = at::vec::fmadd(a1, b1_1, c01);
            c02 = at::vec::fmadd(a1, b2_1, c02);
            c03 = at::vec::fmadd(a1, b3_1, c03);

            a0 = Vec(static_cast<float>(a_base[K + k]));
            c10 = at::vec::fmadd(a0, b0_0, c10);
            c11 = at::vec::fmadd(a0, b1_0, c11);
            c12 = at::vec::fmadd(a0, b2_0, c12);
            c13 = at::vec::fmadd(a0, b3_0, c13);
            a1 = Vec(static_cast<float>(a_base[K + k + 1]));
            c10 = at::vec::fmadd(a1, b0_1, c10);
            c11 = at::vec::fmadd(a1, b1_1, c11);
            c12 = at::vec::fmadd(a1, b2_1, c12);
            c13 = at::vec::fmadd(a1, b3_1, c13);

            a0 = Vec(static_cast<float>(a_base[2 * K + k]));
            c20 = at::vec::fmadd(a0, b0_0, c20);
            c21 = at::vec::fmadd(a0, b1_0, c21);
            c22 = at::vec::fmadd(a0, b2_0, c22);
            c23 = at::vec::fmadd(a0, b3_0, c23);
            a1 = Vec(static_cast<float>(a_base[2 * K + k + 1]));
            c20 = at::vec::fmadd(a1, b0_1, c20);
            c21 = at::vec::fmadd(a1, b1_1, c21);
            c22 = at::vec::fmadd(a1, b2_1, c22);
            c23 = at::vec::fmadd(a1, b3_1, c23);

            a0 = Vec(static_cast<float>(a_base[3 * K + k]));
            c30 = at::vec::fmadd(a0, b0_0, c30);
            c31 = at::vec::fmadd(a0, b1_0, c31);
            c32 = at::vec::fmadd(a0, b2_0, c32);
            c33 = at::vec::fmadd(a0, b3_0, c33);
            a1 = Vec(static_cast<float>(a_base[3 * K + k + 1]));
            c30 = at::vec::fmadd(a1, b0_1, c30);
            c31 = at::vec::fmadd(a1, b1_1, c31);
            c32 = at::vec::fmadd(a1, b2_1, c32);
            c33 = at::vec::fmadd(a1, b3_1, c33);
        }
        for (; k < k_end; ++k) {
            const Vec b0 = Vec::loadu(b_panel + k * block_n);
            const Vec b1 = Vec::loadu(b_panel + k * block_n + vec_size);
            const Vec b2 = Vec::loadu(b_panel + k * block_n + 2 * vec_size);
            const Vec b3 = Vec::loadu(b_panel + k * block_n + 3 * vec_size);
            Vec a(static_cast<float>(a_base[k]));
            c00 = at::vec::fmadd(a, b0, c00);
            c01 = at::vec::fmadd(a, b1, c01);
            c02 = at::vec::fmadd(a, b2, c02);
            c03 = at::vec::fmadd(a, b3, c03);
            a = Vec(static_cast<float>(a_base[K + k]));
            c10 = at::vec::fmadd(a, b0, c10);
            c11 = at::vec::fmadd(a, b1, c11);
            c12 = at::vec::fmadd(a, b2, c12);
            c13 = at::vec::fmadd(a, b3, c13);
            a = Vec(static_cast<float>(a_base[2 * K + k]));
            c20 = at::vec::fmadd(a, b0, c20);
            c21 = at::vec::fmadd(a, b1, c21);
            c22 = at::vec::fmadd(a, b2, c22);
            c23 = at::vec::fmadd(a, b3, c23);
            a = Vec(static_cast<float>(a_base[3 * K + k]));
            c30 = at::vec::fmadd(a, b0, c30);
            c31 = at::vec::fmadd(a, b1, c31);
            c32 = at::vec::fmadd(a, b2, c32);
            c33 = at::vec::fmadd(a, b3, c33);
        }
    }

    c00.store(acc_base);
    c01.store(acc_base + vec_size);
    c02.store(acc_base + 2 * vec_size);
    c03.store(acc_base + 3 * vec_size);
    c10.store(acc_base + block_n);
    c11.store(acc_base + block_n + vec_size);
    c12.store(acc_base + block_n + 2 * vec_size);
    c13.store(acc_base + block_n + 3 * vec_size);
    c20.store(acc_base + 2 * block_n);
    c21.store(acc_base + 2 * block_n + vec_size);
    c22.store(acc_base + 2 * block_n + 2 * vec_size);
    c23.store(acc_base + 2 * block_n + 3 * vec_size);
    c30.store(acc_base + 3 * block_n);
    c31.store(acc_base + 3 * block_n + vec_size);
    c32.store(acc_base + 3 * block_n + 2 * vec_size);
    c33.store(acc_base + 3 * block_n + 3 * vec_size);
}

template <typename T, int64_t rows, int64_t col_vectors, bool accumulate>
inline void gemm_microkernel(
        const T *a_base,
        const typename PackedBType<T>::type *b_panel,
        float *acc_base,
        int64_t K,
        int64_t block_n,
        int64_t k_begin,
        int64_t k_end) {
    constexpr int64_t num_acc = rows * col_vectors;
    constexpr int64_t vec_size = Vec::size();
    std::array<Vec, num_acc> acc;
    std::array<Vec, col_vectors> b_vecs;
    Vec a_vec;

    static_for<0, num_acc>([&](auto index) {
        constexpr int64_t row = index / col_vectors;
        constexpr int64_t col = index % col_vectors;
        acc[index] = load_accumulator<accumulate>(acc_base + row * block_n + col * vec_size);
    });

    using PackedT = typename PackedBType<T>::type;
    if constexpr (std::is_same_v<PackedT, c10::BFloat16>) {
#if defined(__AVX512BF16__)
        const int64_t p_begin = k_begin / 2;
        const int64_t p_end = (k_end + 1) / 2;

        std::array<__m512, num_acc> acc_v;
        static_for<0, num_acc>([&](auto index) {
            acc_v[index] = acc[index].values;
        });

        std::array<__m512, col_vectors> b_v;
        const PackedT *b_ptr = b_panel + p_begin * 2 * block_n;
        std::array<const T*, rows> a_ptrs;
        static_for<0, rows>([&](auto row) {
            a_ptrs[row] = a_base + row * K + 2 * p_begin;
        });
        typedef int32_t __attribute__((__may_alias__)) aliased_int32_t;

        for (int64_t p = p_begin; p < p_end; ++p) {
            static_for<0, col_vectors>([&](auto col) {
                b_v[col] = _mm512_load_ps(reinterpret_cast<const float*>(b_ptr + col * vec_size * 2));
            });

            static_for<0, rows>([&](auto row) {
                __m512 a_v = _mm512_castsi512_ps(_mm512_set1_epi32(*reinterpret_cast<const aliased_int32_t*>(a_ptrs[row])));
                static_for<0, col_vectors>([&](auto col) {
                    constexpr int64_t index = row * col_vectors + col;
                    acc_v[index] = _mm512_dpbf16_ps(acc_v[index], (__m512bh)a_v, (__m512bh)b_v[col]);
                });
            });

            b_ptr += 2 * block_n;
            static_for<0, rows>([&](auto row) {
                a_ptrs[row] += 2;
            });
        }

        static_for<0, num_acc>([&](auto index) {
            acc[index] = Vec(acc_v[index]);
        });
#elif defined(CODA_CPU_ATEN_VEC_ISA_SVE2_BF16)
        aten_vec_sve_bf16::gemm_microkernel<T, rows, col_vectors, accumulate>(
                a_base, b_panel, acc_base, K, block_n, k_begin, k_end);
        return;
#else
        TORCH_CHECK(false, "BFloat16 packed GEMM requires AVX512_BF16 or AArch64 SVE BF16 support");
#endif
    } else {
        for (int64_t k = k_begin; k < k_end; ++k) {
            static_for<0, num_acc>([&](auto index) {
                constexpr int64_t row = index / col_vectors;
                constexpr int64_t col = index % col_vectors;
                if constexpr (col == 0) {
                    a_vec = Vec(static_cast<float>(a_base[row * K + k]));
                }
                if constexpr (row == 0) {
                    b_vecs[col] = Vec::loadu(b_panel + k * block_n + col * vec_size);
                }
                acc[index] = at::vec::fmadd(a_vec, b_vecs[col], acc[index]);
            });
        }
    }

    static_for<0, num_acc>([&](auto index) {
        constexpr int64_t row = index / col_vectors;
        constexpr int64_t col = index % col_vectors;
        acc[index].store(acc_base + row * block_n + col * vec_size);
    });
}

template <typename T, int64_t col_vectors>
inline void gemm_microkernel_rows(
        int64_t rows,
        bool accumulate,
        const T *a_base,
        const typename PackedBType<T>::type *b_panel,
        float *acc_base,
        int64_t K,
        int64_t block_n,
        int64_t k_begin,
        int64_t k_end) {
#define CODA_CALL_MICROKERNEL(ROWS) \
    if (accumulate) { \
        gemm_microkernel<T, ROWS, col_vectors, true>( \
                a_base, b_panel, acc_base, K, block_n, k_begin, k_end); \
    } else { \
        gemm_microkernel<T, ROWS, col_vectors, false>( \
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
                    gemm_microkernel_4x2<T, true>(
                            a_base, b_panel, acc_base, K, k_begin, k_end);
                } else {
                    gemm_microkernel_4x2<T, false>(
                            a_base, b_panel, acc_base, K, k_begin, k_end);
                }
            } else if constexpr (col_vectors == 3) {
                if (accumulate) {
                    gemm_microkernel_4x3<T, true>(
                            a_base, b_panel, acc_base, K, k_begin, k_end);
                } else {
                    gemm_microkernel_4x3<T, false>(
                            a_base, b_panel, acc_base, K, k_begin, k_end);
                }
            } else if constexpr (col_vectors == 4) {
                if (accumulate) {
                    gemm_microkernel_4x4<T, true>(
                            a_base, b_panel, acc_base, K, k_begin, k_end);
                } else {
                    gemm_microkernel_4x4<T, false>(
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

template <typename T, int64_t col_vectors, typename Visitor>
void visit_gemm_vectors_blocked(
        const at::Tensor &A,
        const at::Tensor &B,
        const Visitor &visitor,
        int64_t requested_group_blocks = 0) {
    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);
    const bool use_parallel = (M > 4) || (M * N * K >= 100000);
    const T *a_base = A.data_ptr<T>();
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
                    (kL2Budget / (K * static_cast<int64_t>(sizeof(T))) / row_tile) *
                            row_tile));
    using PackedT = typename PackedBType<T>::type;
    const int64_t raw_kc = kL1Budget / (block_n * static_cast<int64_t>(sizeof(PackedT)));
    const int64_t kc = !use_parallel ? K : std::min<int64_t>(
            K,
            std::max<int64_t>(64, (raw_kc / 64) * 64));
    const auto packed_B = pack_b_register_blocks<T, col_vectors>(B);
    const PackedT *packed_base = packed_B.template data_ptr<PackedT>();

    const int64_t K_padded = std::is_same_v<PackedT, c10::BFloat16> ? (ceil_div(K, 2) * 2) : K;
    const int64_t b_panel_stride = K_padded * block_n;

    if (!use_parallel) {
        alignas(64) std::array<float, kMaxMcRows * col_vectors * vec_size> acc_buffer;
        for (int64_t m0 = 0; m0 < M; m0 += mc_rows) {
            const int64_t rows = std::min<int64_t>(mc_rows, M - m0);
            for (int64_t nb = 0; nb < num_blocks; ++nb) {
                const PackedT *b_panel = packed_base + nb * b_panel_stride;
                for (int64_t k0 = 0; k0 < K; k0 += kc) {
                    const int64_t k_end = std::min<int64_t>(K, k0 + kc);
                    for (int64_t r0 = 0; r0 < rows; r0 += row_tile) {
                        const int64_t micro_rows = std::min<int64_t>(row_tile, rows - r0);
                        gemm_microkernel_rows<T, col_vectors>(
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
    } else if (num_groups > 1) {
        #pragma omp parallel
        {
            alignas(64) std::array<float, kMaxMcRows * col_vectors * vec_size> acc_buffer;
            #pragma omp for schedule(static)
            for (int64_t g_idx = 0; g_idx < num_groups; ++g_idx) {
                const int64_t block_begin = g_idx * group_blocks;
                const int64_t block_end = std::min<int64_t>(num_blocks, block_begin + group_blocks);
                for (int64_t nb = block_begin; nb < block_end; ++nb) {
                    const PackedT *b_panel = packed_base + nb * b_panel_stride;
                    for (int64_t m0 = 0; m0 < M; m0 += mc_rows) {
                        const int64_t rows = std::min<int64_t>(mc_rows, M - m0);
                        for (int64_t k0 = 0; k0 < K; k0 += kc) {
                            const int64_t k_end = std::min<int64_t>(K, k0 + kc);
                            for (int64_t r0 = 0; r0 < rows; r0 += row_tile) {
                                const int64_t micro_rows = std::min<int64_t>(row_tile, rows - r0);
                                gemm_microkernel_rows<T, col_vectors>(
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
        }
    } else {
        const int64_t dynamic_mc_rows = std::max<int64_t>(
                row_tile,
                std::min<int64_t>(
                        mc_rows,
                        (ceil_div(M, at::get_num_threads()) / row_tile) * row_tile));
        const int64_t m_blocks = ceil_div(M, dynamic_mc_rows);
        #pragma omp parallel
        {
            alignas(64) std::array<float, kMaxMcRows * col_vectors * vec_size> acc_buffer;
            #pragma omp for schedule(static)
            for (int64_t m_idx = 0; m_idx < m_blocks; ++m_idx) {
                const int64_t m0 = m_idx * dynamic_mc_rows;
                const int64_t rows = std::min<int64_t>(dynamic_mc_rows, M - m0);
                for (int64_t nb = 0; nb < num_blocks; ++nb) {
                    const PackedT *b_panel = packed_base + nb * b_panel_stride;
                    for (int64_t k0 = 0; k0 < K; k0 += kc) {
                        const int64_t k_end = std::min<int64_t>(K, k0 + kc);
                        for (int64_t r0 = 0; r0 < rows; r0 += row_tile) {
                            const int64_t micro_rows = std::min<int64_t>(row_tile, rows - r0);
                            gemm_microkernel_rows<T, col_vectors>(
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
    }
}

template <typename T, typename Visitor>
void visit_gemm_vectors(
        const at::Tensor &A,
        const at::Tensor &B,
        const Visitor &visitor) {
    visit_gemm_vectors_blocked<T, 4>(A, B, visitor);
}

template <typename T, typename Visitor>
void visit_gemm_vectors_reduction(
        const at::Tensor &A,
        const at::Tensor &B,
        int64_t reduction_block_size,
        const Visitor &visitor) {
    constexpr int64_t block_n = 4 * Vec::size();
    const int64_t num_blocks = ceil_div(B.size(1), block_n);
    const int64_t aligned_group_blocks =
            std::lcm(reduction_block_size, block_n) / block_n;
    const int64_t target_group_blocks = ceil_div(num_blocks, at::get_num_threads());
    const int64_t group_blocks = aligned_group_blocks *
            std::max<int64_t>(1, target_group_blocks / aligned_group_blocks);
    visit_gemm_vectors_blocked<T, 4>(A, B, visitor, group_blocks);
}

template <typename T>
at::Tensor execute_row_scale(
        const at::Tensor &A,
        const at::Tensor &B,
        const at::Tensor &R) {
    require_gemm_inputs_template<T>(A, B);
    require_vector(R, A.size(0), "R");

    const int64_t M = A.size(0);
    const int64_t N = B.size(1);
    auto D = at::empty({M, N}, A.options().dtype(A.scalar_type()));
    const float *r_ptr = R.data_ptr<float>();
    T *d_ptr = D.data_ptr<T>();

    visit_gemm_vectors<T>(A, B, [&](int64_t m, int64_t n, int64_t count, const float *values) {
        T *d_row = d_ptr + m * N;
        const Vec scale(r_ptr[m]);
        for (int64_t i = 0; i < count; i += Vec::size()) {
            const int64_t width = std::min<int64_t>(Vec::size(), count - i);
            const Vec value = Vec::loadu(values + i, width) * scale;
            store_float_as_vec(d_row + n + i, value, width);
        }
    });

    return D;
}

template <typename T>
std::pair<at::Tensor, at::Tensor> execute_swiglu(
        const at::Tensor &A,
        const at::Tensor &B,
        const at::Tensor *R) {
    require_gemm_inputs_template<T>(A, B);
    if (R != nullptr) {
        require_vector(*R, A.size(0), "R");
    }
    TORCH_CHECK(B.size(1) % 2 == 0, "SwiGLU expects an even last dimension");

    const int64_t M = A.size(0);
    const int64_t N = B.size(1);
    auto D = at::empty({M, N}, A.options().dtype(A.scalar_type()));
    auto O = at::empty({M, N / 2}, A.options().dtype(A.scalar_type()));
    const float *r_ptr = R == nullptr ? nullptr : R->data_ptr<float>();
    T *d_ptr = D.data_ptr<T>();
    T *o_ptr = O.data_ptr<T>();

    visit_gemm_vectors<T>(A, B, [&](int64_t m, int64_t n, int64_t count, const float *values) {
        T *d_row = d_ptr + m * N;
        T *o_row = o_ptr + m * (N / 2);
        int64_t i = 0;
        if (r_ptr == nullptr) {
            for (; i + 2 * Vec::size() <= count; i += 2 * Vec::size()) {
                const Vec value0 = Vec::loadu(values + i);
                const Vec value1 = Vec::loadu(values + i + Vec::size());
                store_float_as_vec(d_row + n + i, value0);
                store_float_as_vec(d_row + n + i + Vec::size(), value1);
                const auto gate_up = at::vec::deinterleave2(value0, value1);
                const Vec output =
                        gate_up.first / (Vec(1.0f) + (-gate_up.first).exp()) *
                        gate_up.second;
                store_float_as_vec(o_row + (n + i) / 2, output);
            }
            if (i < count) {
                const int64_t width = count - i;
                const Vec value = Vec::loadu(values + i, width);
                store_float_as_vec(d_row + n + i, value, width);
                const auto gate_up = at::vec::deinterleave2(value, Vec(0.0f));
                const Vec output =
                        gate_up.first / (Vec(1.0f) + (-gate_up.first).exp()) *
                        gate_up.second;
                store_float_as_vec(o_row + (n + i) / 2, output, width / 2);
            }
        } else {
            const Vec scale(r_ptr[m]);
            for (; i + 2 * Vec::size() <= count; i += 2 * Vec::size()) {
                const Vec value0 = Vec::loadu(values + i) * scale;
                const Vec value1 = Vec::loadu(values + i + Vec::size()) * scale;
                store_float_as_vec(d_row + n + i, value0);
                store_float_as_vec(d_row + n + i + Vec::size(), value1);
                const auto gate_up = at::vec::deinterleave2(value0, value1);
                const Vec output =
                        gate_up.first / (Vec(1.0f) + (-gate_up.first).exp()) *
                        gate_up.second;
                store_float_as_vec(o_row + (n + i) / 2, output);
            }
            if (i < count) {
                const int64_t width = count - i;
                const Vec value = Vec::loadu(values + i, width) * scale;
                store_float_as_vec(d_row + n + i, value, width);
                const auto gate_up = at::vec::deinterleave2(value, Vec(0.0f));
                const Vec output =
                        gate_up.first / (Vec(1.0f) + (-gate_up.first).exp()) *
                        gate_up.second;
                store_float_as_vec(o_row + (n + i) / 2, output, width / 2);
            }
        }
    });

    return {D, O};
}

template <typename T>
std::pair<at::Tensor, at::Tensor> execute_rope(
        const at::Tensor &A,
        const at::Tensor &B,
        const at::Tensor *R,
        const at::Tensor &cos_sin,
        bool backward) {
    require_gemm_inputs_template<T>(A, B);
    if (R != nullptr) {
        require_vector(*R, A.size(0), "R");
    }
    require_matrix_template<T>(cos_sin, "cos_sin");
    TORCH_CHECK(cos_sin.size(0) == A.size(0) && cos_sin.size(1) == B.size(1), "RoPE cos_sin shape must match accumulator shape");
    TORCH_CHECK(B.size(1) % 2 == 0, "RoPE expects an even last dimension");

    const int64_t M = A.size(0);
    const int64_t N = B.size(1);
    auto D = at::empty({M, N}, A.options().dtype(A.scalar_type()));
    auto O = at::empty({M, N}, A.options().dtype(A.scalar_type()));
    const float *r_ptr = R == nullptr ? nullptr : R->data_ptr<float>();
    const T *cs_ptr = cos_sin.data_ptr<T>();
    T *d_ptr = D.data_ptr<T>();
    T *o_ptr = O.data_ptr<T>();
    const float sign = backward ? -1.0f : 1.0f;

    visit_gemm_vectors<T>(A, B, [&](int64_t m, int64_t n, int64_t count, const float *values) {
        T *d_row = d_ptr + m * N;
        T *o_row = o_ptr + m * N;
        const T *cs_row = cs_ptr + m * N;
        const Vec sign_vec(sign);
        int64_t i = 0;
        if (r_ptr == nullptr) {
            for (; i + 2 * Vec::size() <= count; i += 2 * Vec::size()) {
                const Vec value0 = Vec::loadu(values + i);
                const Vec value1 = Vec::loadu(values + i + Vec::size());
                store_float_as_vec(d_row + n + i, value0);
                store_float_as_vec(d_row + n + i + Vec::size(), value1);
                const auto x = at::vec::deinterleave2(value0, value1);
                const auto cs = at::vec::deinterleave2(
                        load_vec_as_float(cs_row + n + i),
                        load_vec_as_float(cs_row + n + i + Vec::size()));
                const Vec s = cs.second * sign_vec;
                const Vec y0 = x.first * cs.first + x.second * s;
                const Vec y1 = x.first * (-s) + x.second * cs.first;
                const auto output = at::vec::interleave2(y0, y1);
                store_float_as_vec(o_row + n + i, output.first);
                store_float_as_vec(o_row + n + i + Vec::size(), output.second);
            }
            if (i < count) {
                const int64_t width = count - i;
                const Vec value = Vec::loadu(values + i, width);
                store_float_as_vec(d_row + n + i, value, width);
                const auto x = at::vec::deinterleave2(value, Vec(0.0f));
                const auto cs = at::vec::deinterleave2(
                        load_vec_as_float(cs_row + n + i, width),
                        Vec(0.0f));
                const Vec s = cs.second * sign_vec;
                const Vec y0 = x.first * cs.first + x.second * s;
                const Vec y1 = x.first * (-s) + x.second * cs.first;
                store_float_as_vec(o_row + n + i, at::vec::interleave2(y0, y1).first, width);
            }
        } else {
            const Vec scale(r_ptr[m]);
            for (; i + 2 * Vec::size() <= count; i += 2 * Vec::size()) {
                const Vec value0 = Vec::loadu(values + i) * scale;
                const Vec value1 = Vec::loadu(values + i + Vec::size()) * scale;
                store_float_as_vec(d_row + n + i, value0);
                store_float_as_vec(d_row + n + i + Vec::size(), value1);
                const auto x = at::vec::deinterleave2(value0, value1);
                const auto cs = at::vec::deinterleave2(
                        load_vec_as_float(cs_row + n + i),
                        load_vec_as_float(cs_row + n + i + Vec::size()));
                const Vec s = cs.second * sign_vec;
                const Vec y0 = x.first * cs.first + x.second * s;
                const Vec y1 = x.first * (-s) + x.second * cs.first;
                const auto output = at::vec::interleave2(y0, y1);
                store_float_as_vec(o_row + n + i, output.first);
                store_float_as_vec(o_row + n + i + Vec::size(), output.second);
            }
            if (i < count) {
                const int64_t width = count - i;
                const Vec value = Vec::loadu(values + i, width) * scale;
                store_float_as_vec(d_row + n + i, value, width);
                const auto x = at::vec::deinterleave2(value, Vec(0.0f));
                const auto cs = at::vec::deinterleave2(
                        load_vec_as_float(cs_row + n + i, width),
                        Vec(0.0f));
                const Vec s = cs.second * sign_vec;
                const Vec y0 = x.first * cs.first + x.second * s;
                const Vec y1 = x.first * (-s) + x.second * cs.first;
                store_float_as_vec(o_row + n + i, at::vec::interleave2(y0, y1).first, width);
            }
        }
    });

    return {D, O};
}

template <typename T>
std::tuple<at::Tensor, at::Tensor, at::Tensor> execute_residual_partial_rmsnorm(
        const at::Tensor &A,
        const at::Tensor &B,
        const at::Tensor &C,
        const at::Tensor &W,
        int64_t block_size) {
    require_gemm_inputs_template<T>(A, B);
    require_matrix_template<T>(C, "C");
    require_vector_template<T>(W, B.size(1), "W");
    TORCH_CHECK(C.size(0) == A.size(0) && C.size(1) == B.size(1), "residual shape mismatch");
    TORCH_CHECK(block_size > 0, "block post-op requires a positive block_size");
    TORCH_CHECK(B.size(1) % block_size == 0, "N must be divisible by block_size");

    const int64_t M = A.size(0);
    const int64_t N = B.size(1);
    const int64_t num_blocks = N / block_size;
    auto D = at::empty({M, N}, A.options().dtype(A.scalar_type()));
    auto S_float = at::zeros({M, num_blocks}, A.options().dtype(at::kFloat));
    auto O = at::empty({M, N}, A.options().dtype(A.scalar_type()));
    const T *c_base = C.data_ptr<T>();
    const T *w_ptr = W.data_ptr<T>();
    T *d_ptr = D.data_ptr<T>();
    float *s_float_ptr = S_float.data_ptr<float>();
    T *o_ptr = O.data_ptr<T>();

    visit_gemm_vectors_reduction<T>(A, B, block_size, [&](int64_t m, int64_t n, int64_t count, const float *values) {
        T *d_row = d_ptr + m * N;
        float *s_row = s_float_ptr + m * num_blocks;
        T *o_row = o_ptr + m * N;
        const T *c_row = c_base + m * N;
        for (int64_t i = 0; i < count; i += Vec::size()) {
            const int64_t width = std::min<int64_t>(Vec::size(), count - i);
            const int64_t col = n + i;
            const Vec value =
                    Vec::loadu(values + i, width) + load_vec_as_float(c_row + col, width);
            store_float_as_vec(d_row + col, value, width);
            store_float_as_vec(o_row + col, value * load_vec_as_float(w_ptr + col, width), width);
            if (col / block_size == (col + width - 1) / block_size) {
                const Vec square = value * value;
                float sum_sq = 0.0f;
                if (width == Vec::size()) {
                    sum_sq = vec_reduce_sum(square);
                } else {
                    alignas(64) std::array<float, Vec::size()> scalar_values;
                    square.store(scalar_values.data(), width);
                    for (int64_t j = 0; j < width; ++j) {
                        sum_sq += scalar_values[j];
                    }
                }
                s_row[col / block_size] += sum_sq;
            } else {
                alignas(64) std::array<float, Vec::size()> scalar_values;
                value.store(scalar_values.data(), width);
                for (int64_t j = 0; j < width; ++j) {
                    const float scalar = scalar_values[j];
                    s_row[(col + j) / block_size] += (scalar * scalar);
                }
            }
        }
    });

    S_float.div_(static_cast<float>(block_size));
    return {D, S_float, O};
}

template <typename T>
std::tuple<at::Tensor, at::Tensor, at::Tensor> execute_partial_cross_entropy(
        const at::Tensor &A,
        const at::Tensor &B,
        const at::Tensor *R,
        const at::Tensor &targets,
        int64_t block_size) {
    require_gemm_inputs_template<T>(A, B);
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
    auto logits = at::empty({M, N}, A.options().dtype(A.scalar_type()));
    auto logits_tgt = at::empty({M}, A.options().dtype(A.scalar_type()));
    auto logits_lse = at::empty({M, num_blocks}, A.options().dtype(at::kFloat));
    const T *a_base = A.data_ptr<T>();
    const T *b_base = B.data_ptr<T>();
    const float *r_ptr = R == nullptr ? nullptr : R->data_ptr<float>();
    const int64_t *targets_ptr = targets.data_ptr<int64_t>();
    T *logits_ptr = logits.data_ptr<T>();
    T *logits_tgt_ptr = logits_tgt.data_ptr<T>();
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
                        const Vec a_vec(static_cast<float>(a_base[m * K + k]));
                        const Vec b_vec = load_vec_as_float<T>(b_base + k * N + n, count);
                        acc = at::vec::fmadd(a_vec, b_vec, acc);
                    }
                    acc.store(values, count);
                    T *logits_row = logits_ptr + m * N;
                    for (int64_t i = 0; i < count; ++i) {
                        const int64_t col = n + i;
                        const float value = values[i] * scale;
                        logits_row[col] = static_cast<T>(value);
                        if (col == target) {
                            logits_tgt_ptr[m] = static_cast<T>(value);
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

template <typename T>
std::pair<at::Tensor, py::dict> execute_aten_vec_postops_template(
        const std::vector<PostOpNode> &parsed_nodes,
        const TensorMap &tensor_map,
        const at::Tensor &A,
        const at::Tensor &B) {
    py::dict side_outputs;

    if (is_residual_partial_rmsnorm(parsed_nodes)) {
        const auto &C = require_tensor(tensor_map, require_name(parsed_nodes[0].tensor, "tensor"));
        const auto &W = require_tensor(tensor_map, require_name(parsed_nodes[2].tensor, "tensor"));
        auto outputs = execute_residual_partial_rmsnorm<T>(A, B, C, W, parsed_nodes[1].block_size);
        side_outputs[py::str(require_name(parsed_nodes[1].output, "output"))] = std::get<1>(outputs);
        side_outputs[py::str(require_name(parsed_nodes[2].output, "output"))] = std::get<2>(outputs);
        return {std::get<0>(outputs), side_outputs};
    }

    if (is_row_scale(parsed_nodes)) {
        const auto &R = require_tensor(tensor_map, require_name(parsed_nodes[0].tensor, "tensor"));
        return {execute_row_scale<T>(A, B, R), side_outputs};
    }

    if (is_swiglu(parsed_nodes)) {
        auto outputs = execute_swiglu<T>(A, B, nullptr);
        side_outputs[py::str(require_name(parsed_nodes[0].output, "output"))] = outputs.second;
        return {outputs.first, side_outputs};
    }

    if (is_row_scale_swiglu(parsed_nodes)) {
        const auto &R = require_tensor(tensor_map, require_name(parsed_nodes[0].tensor, "tensor"));
        auto outputs = execute_swiglu<T>(A, B, &R);
        side_outputs[py::str(require_name(parsed_nodes[1].output, "output"))] = outputs.second;
        return {outputs.first, side_outputs};
    }

    if (is_rope(parsed_nodes)) {
        const auto &cos_sin = require_tensor(tensor_map, require_name(parsed_nodes[0].tensor, "tensor"));
        auto outputs = execute_rope<T>(A, B, nullptr, cos_sin, parsed_nodes[0].backward);
        side_outputs[py::str(require_name(parsed_nodes[0].output, "output"))] = outputs.second;
        return {outputs.first, side_outputs};
    }

    if (is_row_scale_rope(parsed_nodes)) {
        const auto &R = require_tensor(tensor_map, require_name(parsed_nodes[0].tensor, "tensor"));
        const auto &cos_sin = require_tensor(tensor_map, require_name(parsed_nodes[1].tensor, "tensor"));
        auto outputs = execute_rope<T>(A, B, &R, cos_sin, parsed_nodes[1].backward);
        side_outputs[py::str(require_name(parsed_nodes[1].output, "output"))] = outputs.second;
        return {outputs.first, side_outputs};
    }

    if (is_partial_cross_entropy(parsed_nodes)) {
        const auto &targets = require_tensor(tensor_map, require_name(parsed_nodes[0].tensor, "tensor"));
        auto outputs = execute_partial_cross_entropy<T>(A, B, nullptr, targets, parsed_nodes[1].block_size);
        side_outputs[py::str(require_name(parsed_nodes[0].output, "output"))] = std::get<1>(outputs);
        side_outputs[py::str(require_name(parsed_nodes[1].output, "output"))] = std::get<2>(outputs);
        return {std::get<0>(outputs), side_outputs};
    }

    if (is_row_scale_partial_cross_entropy(parsed_nodes)) {
        const auto &R = require_tensor(tensor_map, require_name(parsed_nodes[0].tensor, "tensor"));
        const auto &targets = require_tensor(tensor_map, require_name(parsed_nodes[1].tensor, "tensor"));
        auto outputs = execute_partial_cross_entropy<T>(A, B, &R, targets, parsed_nodes[2].block_size);
        side_outputs[py::str(require_name(parsed_nodes[1].output, "output"))] = std::get<1>(outputs);
        side_outputs[py::str(require_name(parsed_nodes[2].output, "output"))] = std::get<2>(outputs);
        return {std::get<0>(outputs), side_outputs};
    }

    throw std::invalid_argument("unsupported ATen vector post-op program");
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
#if defined(CODA_CPU_ATEN_VEC_ISA_AVX512)
    return "avx512";
#elif defined(CODA_CPU_ATEN_VEC_ISA_AVX2)
    return "avx2";
#elif defined(CODA_CPU_ATEN_VEC_ISA_SVE2_BF16)
    return "sve2-bf16";
#elif defined(CODA_CPU_ATEN_VEC_ISA_SVE256)
    return "sve256";
#elif defined(CODA_CPU_ATEN_VEC_ISA_GENERIC)
#if defined(__aarch64__) || defined(_M_ARM64)
    return "neon";
#else
    return "generic";
#endif
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

    if (A.scalar_type() == at::kFloat) {
        return execute_aten_vec_postops_template<float>(parsed_nodes, tensor_map, A, B);
    } else if (A.scalar_type() == at::kBFloat16) {
        return execute_aten_vec_postops_template<c10::BFloat16>(parsed_nodes, tensor_map, A, B);
    } else {
        TORCH_CHECK(false, "aten-vec provider only supports float32 and bfloat16 dtypes");
    }
#endif
}

void prepack_weight(const at::Tensor &B) {
#if defined(CODA_CPU_WITH_ATEN_VEC)
    if (B.scalar_type() == at::kFloat) {
        pack_b_register_blocks<float, 4>(B);
    } else if (B.scalar_type() == at::kBFloat16) {
        pack_b_register_blocks<c10::BFloat16, 4>(B);
    }
#endif
}


#if defined(CODA_CPU_WITH_ATEN_VEC)
template <typename T>
at::Tensor execute_gemm_pure(const at::Tensor &A, const at::Tensor &B) {
    require_gemm_inputs_template<T>(A, B);
    const int64_t M = A.size(0);
    const int64_t N = B.size(1);
    auto D = at::empty({M, N}, A.options().dtype(A.scalar_type()));
    T *d_ptr = D.data_ptr<T>();
    visit_gemm_vectors<T>(A, B, [&](int64_t m, int64_t n, int64_t count, const float *values) {
        T *d_row = d_ptr + m * N;
        for (int64_t i = 0; i < count; i += Vec::size()) {
            const int64_t width = std::min<int64_t>(Vec::size(), count - i);
            const Vec value = Vec::loadu(values + i, width);
            store_float_as_vec(d_row + n + i, value, width);
        }
    });
    return D;
}

template <typename T>
void apply_rope_half_one_vector(
    T *out,
    const T *in,
    const T *cos_ptr,
    const T *sin_ptr,
    int64_t head_dim
) {
    const int64_t half = head_dim / 2;
    int64_t i = 0;
    for (; i <= half - Vec::size(); i += Vec::size()) {
        Vec x1 = load_vec_as_float(in + i);
        Vec x2 = load_vec_as_float(in + i + half);
        Vec c1 = load_vec_as_float(cos_ptr + i);
        Vec s1 = load_vec_as_float(sin_ptr + i);
        Vec y1 = x1 * c1 - x2 * s1;
        store_float_as_vec(out + i, y1);
        Vec c2 = load_vec_as_float(cos_ptr + i + half);
        Vec s2 = load_vec_as_float(sin_ptr + i + half);
        Vec y2 = x2 * c2 + x1 * s2;
        store_float_as_vec(out + i + half, y2);
    }
    for (; i < half; ++i) {
        float x1 = static_cast<float>(in[i]);
        float x2 = static_cast<float>(in[i + half]);
        float c1 = static_cast<float>(cos_ptr[i]);
        float s1 = static_cast<float>(sin_ptr[i]);
        out[i] = static_cast<T>(x1 * c1 - x2 * s1);
        float c2 = static_cast<float>(cos_ptr[i + half]);
        float s2 = static_cast<float>(sin_ptr[i + half]);
        out[i + half] = static_cast<T>(x2 * c2 + x1 * s2);
    }
}

template <typename T>
at::Tensor apply_rmsnorm(const at::Tensor &x, const at::Tensor &w, double eps) {
    const int64_t N = w.size(0);
    const int64_t M = x.numel() / N;
    auto y = at::empty_like(x);
    const T *x_ptr = x.data_ptr<T>();
    const T *w_ptr = w.data_ptr<T>();
    T *y_ptr = y.data_ptr<T>();
    auto worker = [&](int64_t begin, int64_t end) {
        for (int64_t m = begin; m < end; ++m) {
            const T *x_row = x_ptr + m * N;
            T *y_row = y_ptr + m * N;
            float sum_sq = 0.0f;
            int64_t d = 0;
            Vec sum_sq_vec(0.0f);
            for (; d <= N - Vec::size(); d += Vec::size()) {
                Vec xv = load_vec_as_float(x_row + d);
                sum_sq_vec = sum_sq_vec + xv * xv;
            }
            sum_sq = vec_reduce_sum(sum_sq_vec);
            for (; d < N; ++d) {
                float xv = static_cast<float>(x_row[d]);
                sum_sq += xv * xv;
            }
            float rstd = 1.0f / std::sqrt(sum_sq / N + static_cast<float>(eps));
            d = 0;
            Vec rstd_vec(rstd);
            for (; d <= N - Vec::size(); d += Vec::size()) {
                Vec xv = load_vec_as_float(x_row + d);
                Vec wv = load_vec_as_float(w_ptr + d);
                Vec yv = xv * rstd_vec * wv;
                store_float_as_vec(y_row + d, yv);
            }
            for (; d < N; ++d) {
                y_row[d] = static_cast<T>(static_cast<float>(x_row[d]) * rstd * static_cast<float>(w_ptr[d]));
            }
        }
    };
    if (M > 1) {
        at::parallel_for(0, M, 1, worker);
    } else {
        worker(0, M);
    }
    return y;
}

template <typename T>
at::Tensor split_transpose_rope_cache_template(
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
    double rms_norm_eps
) {
    const int64_t B = qkv.size(0);
    const int64_t T_seq = qkv.size(1);
    auto Q_out = at::empty({B, num_heads, T_seq, head_dim}, qkv.options());
    const T *qkv_ptr = qkv.data_ptr<T>();
    const T *cos_ptr = cos.data_ptr<T>();
    const T *sin_ptr = sin.data_ptr<T>();
    T *q_out_ptr = Q_out.data_ptr<T>();
    T *k_cache_ptr = k_cache.data_ptr<T>();
    T *v_cache_ptr = v_cache.data_ptr<T>();
    const int64_t k_layer_stride = k_cache.stride(0);
    const int64_t k_batch_stride = k_cache.stride(1);
    const int64_t k_kv_head_stride = k_cache.stride(2);
    const int64_t k_seq_stride = k_cache.stride(3);
    const int64_t k_dim_stride = k_cache.stride(4);
    const int64_t v_layer_stride = v_cache.stride(0);
    const int64_t v_batch_stride = v_cache.stride(1);
    const int64_t v_kv_head_stride = v_cache.stride(2);
    const int64_t v_seq_stride = v_cache.stride(3);
    const int64_t v_dim_stride = v_cache.stride(4);
    const int64_t q_dim = num_heads * head_dim;
    const int64_t k_dim = num_kv_heads * head_dim;
    const int64_t qkv_dim = q_dim + 2 * k_dim;
    auto worker = [&](int64_t begin, int64_t end) {
        std::vector<float> q_norm_buf;
        std::vector<float> k_norm_buf;
        if (is_qwen3) {
            q_norm_buf.resize(q_dim);
            k_norm_buf.resize(k_dim);
        }
        for (int64_t bt = begin; bt < end; ++bt) {
            const int64_t b = bt / T_seq;
            const int64_t t = bt % T_seq;
            const T *src_qkv = qkv_ptr + b * T_seq * qkv_dim + t * qkv_dim;
            const T *src_q = src_qkv;
            const T *src_k = src_qkv + q_dim;
            const T *src_v = src_qkv + q_dim + k_dim;
            const int64_t seq_len_cos = cos.size(cos.dim() - 2);
            const int64_t t_cos = (seq_len_cos == 1) ? 0 : t;
            const T *cos_step = cos_ptr + t_cos * cos.stride(cos.dim() - 2);
            const T *sin_step = sin_ptr + t_cos * sin.stride(sin.dim() - 2);
            if (is_qwen3) {
                const T *qw = q_norm_w.data_ptr<T>();
                for (int64_t h = 0; h < num_heads; ++h) {
                    const T *q_head = src_q + h * head_dim;
                    const T *qw_head = qw;
                    float sum_sq = 0.0f;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        float val = static_cast<float>(q_head[d]);
                        sum_sq += val * val;
                    }
                    float rstd = 1.0f / std::sqrt(sum_sq / head_dim + static_cast<float>(rms_norm_eps));
                    for (int64_t d = 0; d < head_dim; ++d) {
                        q_norm_buf[h * head_dim + d] = static_cast<float>(q_head[d]) * rstd * static_cast<float>(qw_head[d]);
                    }
                }
                const T *kw = k_norm_w.data_ptr<T>();
                for (int64_t h = 0; h < num_kv_heads; ++h) {
                    const T *k_head = src_k + h * head_dim;
                    const T *kw_head = kw;
                    float sum_sq = 0.0f;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        float val = static_cast<float>(k_head[d]);
                        sum_sq += val * val;
                    }
                    float rstd = 1.0f / std::sqrt(sum_sq / head_dim + static_cast<float>(rms_norm_eps));
                    for (int64_t d = 0; d < head_dim; ++d) {
                        k_norm_buf[h * head_dim + d] = static_cast<float>(k_head[d]) * rstd * static_cast<float>(kw_head[d]);
                    }
                }
            }
            for (int64_t h = 0; h < num_heads; ++h) {
                T *dst_q = q_out_ptr + b * num_heads * T_seq * head_dim + h * T_seq * head_dim + t * head_dim;
                alignas(16) T temp_in[512];
                if (is_qwen3) {
                    for (int64_t d = 0; d < head_dim; ++d) {
                        temp_in[d] = static_cast<T>(q_norm_buf[h * head_dim + d]);
                    }
                } else {
                    std::copy_n(src_q + h * head_dim, head_dim, temp_in);
                }
                apply_rope_half_one_vector<T>(dst_q, temp_in, cos_step, sin_step, head_dim);
            }
            for (int64_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
                T *dst_k = k_cache_ptr + layer_idx * k_layer_stride + b * k_batch_stride + kv_h * k_kv_head_stride + (cache_index + t) * k_seq_stride;
                alignas(16) T temp_in[512];
                if (is_qwen3) {
                    for (int64_t d = 0; d < head_dim; ++d) {
                        temp_in[d] = static_cast<T>(k_norm_buf[kv_h * head_dim + d]);
                    }
                } else {
                    std::copy_n(src_k + kv_h * head_dim, head_dim, temp_in);
                }
                alignas(16) T temp_out[512];
                apply_rope_half_one_vector<T>(temp_out, temp_in, cos_step, sin_step, head_dim);
                for (int64_t d = 0; d < head_dim; ++d) {
                    dst_k[d * k_dim_stride] = temp_out[d];
                }
            }
            for (int64_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
                T *dst_v = v_cache_ptr + layer_idx * v_layer_stride + b * v_batch_stride + kv_h * v_kv_head_stride + (cache_index + t) * v_seq_stride;
                const T *src_v_head = src_v + kv_h * head_dim;
                for (int64_t d = 0; d < head_dim; ++d) {
                    dst_v[d * v_dim_stride] = src_v_head[d];
                }
            }
        }
    };
    if (B * T_seq > 1) {
        at::parallel_for(0, B * T_seq, 1, worker);
    } else {
        worker(0, B * T_seq);
    }
    return Q_out;
}

template <typename T>
at::Tensor coda_decode_attention_impl(
    const at::Tensor &Q,
    const at::Tensor &k_cache,
    const at::Tensor &v_cache,
    int64_t layer_idx,
    int64_t seq_len,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim
) {
    const int64_t B = Q.size(0);
    auto Y = at::empty({B, 1, num_heads * head_dim}, Q.options());
    const T *q_ptr = Q.data_ptr<T>();
    const T *k_ptr = k_cache.data_ptr<T>();
    const T *v_ptr = v_cache.data_ptr<T>();
    T *y_ptr = Y.data_ptr<T>();
    const int64_t group_size = num_heads / num_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int64_t k_layer_stride = k_cache.stride(0);
    const int64_t k_batch_stride = k_cache.stride(1);
    const int64_t k_kv_head_stride = k_cache.stride(2);
    const int64_t k_seq_stride = k_cache.stride(3);
    const int64_t k_dim_stride = k_cache.stride(4);
    const int64_t v_layer_stride = v_cache.stride(0);
    const int64_t v_batch_stride = v_cache.stride(1);
    const int64_t v_kv_head_stride = v_cache.stride(2);
    const int64_t v_seq_stride = v_cache.stride(3);
    const int64_t v_dim_stride = v_cache.stride(4);
    const int64_t q_batch_stride = Q.stride(0);
    const int64_t q_head_stride = Q.stride(1);
    const int64_t q_dim_stride = Q.dim() == 4 ? Q.stride(3) : Q.stride(2);
    const int64_t y_batch_stride = num_heads * head_dim;
    const int64_t y_head_stride = head_dim;
    const int64_t y_dim_stride = 1;
    const bool use_parallel = (B * num_heads * seq_len * head_dim) >= 65536;
    auto worker = [&](int64_t begin, int64_t end) {
        float *scores = nullptr;
        std::vector<float> scores_vec;
        if (seq_len < 4096) {
            scores = (float*)alloca(seq_len * sizeof(float));
        } else {
            scores_vec.resize(seq_len);
            scores = scores_vec.data();
        }
        for (int64_t bh = begin; bh < end; ++bh) {
            const int64_t b = bh / num_heads;
            const int64_t h = bh % num_heads;
            const int64_t kv_h = h / group_size;
            const T *q_head_ptr = q_ptr + b * q_batch_stride + h * q_head_stride;
            float max_score = -std::numeric_limits<float>::infinity();
            for (int64_t j = 0; j < seq_len; ++j) {
                const T *k_vec_ptr = k_ptr + layer_idx * k_layer_stride + b * k_batch_stride + kv_h * k_kv_head_stride + j * k_seq_stride;
                float sum = 0.0f;
                int64_t d = 0;
                Vec sum_vec(0.0f);
                for (; d <= head_dim - Vec::size(); d += Vec::size()) {
                    Vec q_v = load_vec_as_float(q_head_ptr + d * q_dim_stride);
                    Vec k_v = load_vec_as_float(k_vec_ptr + d * k_dim_stride);
                    sum_vec = sum_vec + q_v * k_v;
                }
                sum = vec_reduce_sum(sum_vec);
                for (; d < head_dim; ++d) {
                    sum += static_cast<float>(q_head_ptr[d * q_dim_stride]) * static_cast<float>(k_vec_ptr[d * k_dim_stride]);
                }
                float score = sum * scale;
                scores[j] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }
            float sum_exp = 0.0f;
            for (int64_t j = 0; j < seq_len; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                sum_exp += scores[j];
            }
            float inv_sum_exp = 1.0f / sum_exp;
            for (int64_t j = 0; j < seq_len; ++j) {
                scores[j] *= inv_sum_exp;
            }
            T *y_head_ptr = y_ptr + b * y_batch_stride + h * y_head_stride;
            for (int64_t d = 0; d < head_dim; ++d) {
                y_head_ptr[d * y_dim_stride] = static_cast<T>(0.0f);
            }
            for (int64_t j = 0; j < seq_len; ++j) {
                const float sj = scores[j];
                const T *v_vec_ptr = v_ptr + layer_idx * v_layer_stride + b * v_batch_stride + kv_h * v_kv_head_stride + j * v_seq_stride;
                int64_t d = 0;
                Vec sj_vec(sj);
                for (; d <= head_dim - Vec::size(); d += Vec::size()) {
                    Vec y_v = load_vec_as_float(y_head_ptr + d * y_dim_stride);
                    Vec v_v = load_vec_as_float(v_vec_ptr + d * v_dim_stride);
                    y_v = y_v + sj_vec * v_v;
                    store_float_as_vec(y_head_ptr + d * y_dim_stride, y_v);
                }
                for (; d < head_dim; ++d) {
                    y_head_ptr[d * y_dim_stride] = static_cast<T>(static_cast<float>(y_head_ptr[d * y_dim_stride]) + sj * static_cast<float>(v_vec_ptr[d * v_dim_stride]));
                }
            }
        }
    };
    if (use_parallel) {
        at::parallel_for(0, B * num_heads, 1, worker);
    } else {
        worker(0, B * num_heads);
    }
    return Y;
}

template <typename T>
at::Tensor coda_qwen_forward_template(
    CodaQwenModel &model,
    const at::Tensor &input_ids,
    const at::Tensor &cos,
    const at::Tensor &sin,
    at::Tensor &k_cache,
    at::Tensor &v_cache,
    int64_t cache_index
) {
    const int64_t B = input_ids.size(0);
    const int64_t T_seq = input_ids.size(1);
    auto x = at::embedding(model.embed_tokens_weight, input_ids);
    auto input_norm_w = model.input_layernorm_weights[0];
    auto h = apply_rmsnorm<T>(x, input_norm_w, model.rms_norm_eps);
    auto w3 = model.w3_weights[0];
    auto h_2d = h.reshape({B * T_seq, -1});
    auto qkv_2d = at::matmul(h_2d, w3);
    if (model.qkv_biases[0].defined() && model.qkv_biases[0].numel() > 0) {
        qkv_2d.add_(model.qkv_biases[0]);
    }
    auto qkv = qkv_2d.reshape({B, T_seq, -1});
    auto x_current = x;
    for (int64_t l = 0; l < model.num_layers; ++l) {
        at::Tensor q_norm_w = model.q_norm_weights[l];
        at::Tensor k_norm_w = model.k_norm_weights[l];
        auto q = split_transpose_rope_cache_template<T>(
            qkv, cos, sin, k_cache, v_cache, l, cache_index,
            model.num_heads, model.num_kv_heads, model.head_dim, model.is_qwen3,
            q_norm_w, k_norm_w, model.rms_norm_eps
        );
        at::Tensor attn_out;
        if (T_seq > 1) {
            auto k_full = k_cache.select(0, l).slice(2, 0, cache_index + T_seq);
            auto v_full = v_cache.select(0, l).slice(2, 0, cache_index + T_seq);
            if (model.num_heads != model.num_kv_heads) {
                const int64_t group_size = model.num_heads / model.num_kv_heads;
                auto q_5d = q.view({B, model.num_kv_heads, group_size, T_seq, model.head_dim});
                auto k_5d = k_full.unsqueeze(2);
                auto v_5d = v_full.unsqueeze(2);
                auto attn_out_5d = at::scaled_dot_product_attention(q_5d, k_5d, v_5d, {}, 0.0, true);
                attn_out = attn_out_5d.view({B, model.num_heads, T_seq, model.head_dim});
            } else {
                attn_out = at::scaled_dot_product_attention(q, k_full, v_full, {}, 0.0, true);
            }
            attn_out = attn_out.transpose(1, 2).reshape({B, T_seq, -1}).contiguous();
        } else {
            attn_out = coda_decode_attention_impl<T>(
                q, k_cache, v_cache, l, cache_index + 1,
                model.num_heads, model.num_kv_heads, model.head_dim
            );
        }
        auto w0 = model.w0_weights[l];
        auto w1 = model.w1_weights[l];
        auto wn0 = model.post_attention_layernorm_weights[l];
        auto M_size = B * T_seq;
        auto attn_out_2d = attn_out.reshape({M_size, -1});
        auto x_current_2d = x_current.reshape({M_size, -1});
        auto rmsnorm_outputs = execute_residual_partial_rmsnorm<T>(
            attn_out_2d, w0, x_current_2d, wn0, 128
        );
        auto x_mlp_res_2d = std::get<0>(rmsnorm_outputs);
        auto s_mlp = std::get<1>(rmsnorm_outputs);
        auto h_mlp_2d = std::get<2>(rmsnorm_outputs);
        auto rstd_mlp = 1.0f / at::sqrt(s_mlp.mean(1) + model.rms_norm_eps);
        auto swiglu_outputs = execute_swiglu<T>(h_mlp_2d, w1, &rstd_mlp);
        auto y_swiglu_2d = swiglu_outputs.second;
        if (l < model.num_layers - 1) {
            auto w2 = model.w2_weights[l];
            auto w3_next = model.w3_weights[l + 1];
            auto wn1_next = model.input_layernorm_weights[l + 1];
            auto next_outputs = execute_residual_partial_rmsnorm<T>(
                y_swiglu_2d, w2, x_mlp_res_2d, wn1_next, 128
            );
            auto x_next_2d = std::get<0>(next_outputs);
            auto s_next = std::get<1>(next_outputs);
            auto h_next_2d = std::get<2>(next_outputs);
            auto rstd_next = 1.0f / at::sqrt(s_next.mean(1) + model.rms_norm_eps);
            auto qkv_next_2d = execute_row_scale<T>(h_next_2d, w3_next, rstd_next);
            if (model.qkv_biases[l + 1].defined() && model.qkv_biases[l + 1].numel() > 0) {
                qkv_next_2d.add_(model.qkv_biases[l + 1]);
            }
            qkv = qkv_next_2d.reshape({B, T_seq, -1});
            x_current = x_next_2d.reshape({B, T_seq, -1});
        } else {
            auto w2 = model.w2_weights[l];
            auto x_final_2d = x_mlp_res_2d + at::matmul(y_swiglu_2d, w2);
            auto x_final = x_final_2d.reshape({B, T_seq, -1});
            auto h_final = apply_rmsnorm<T>(x_final, model.final_norm_weight, model.rms_norm_eps);
            auto h_final_2d = h_final.reshape({B * T_seq, -1});
            auto logits_2d = at::matmul(h_final_2d, model.lm_head_weight);
            auto logits = logits_2d.reshape({B, T_seq, -1});
            return logits;
        }
    }
    throw std::runtime_error("Unexpected end of layer loop");
}
#endif

CodaQwenModel::CodaQwenModel(
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
) {
#if !defined(CODA_CPU_WITH_ATEN_VEC)
    throw std::runtime_error("ATen vector provider was not compiled into this native extension");
#else
    this->embed_tokens_weight = embed_tokens_weight;
    this->input_layernorm_weights = input_layernorm_weights;
    this->post_attention_layernorm_weights = post_attention_layernorm_weights;
    this->w3_weights = w3_weights;
    this->qkv_biases = qkv_biases;
    this->w0_weights = w0_weights;
    this->w1_weights = w1_weights;
    this->w2_weights = w2_weights;
    this->q_norm_weights = q_norm_weights;
    this->k_norm_weights = k_norm_weights;
    this->final_norm_weight = final_norm_weight;
    this->lm_head_weight = lm_head_weight;
    this->rms_norm_eps = rms_norm_eps;
    this->num_heads = num_heads;
    this->num_kv_heads = num_kv_heads;
    this->head_dim = head_dim;
    this->is_qwen3 = is_qwen3;
    this->num_layers = w3_weights.size();

    // Safely prepack weights in the constructor to ensure cache residency
    for (auto &w : this->w0_weights) prepack_weight(w);
    for (auto &w : this->w1_weights) prepack_weight(w);
    for (auto &w : this->w2_weights) prepack_weight(w);
    for (auto &w : this->w3_weights) prepack_weight(w);
    prepack_weight(this->lm_head_weight);
#endif
}

at::Tensor CodaQwenModel::forward(
    const at::Tensor &input_ids,
    const at::Tensor &cos,
    const at::Tensor &sin,
    at::Tensor &k_cache,
    at::Tensor &v_cache,
    int64_t cache_index
) {
#if !defined(CODA_CPU_WITH_ATEN_VEC)
    throw std::runtime_error("ATen vector provider was not compiled into this native extension");
#else
    if (embed_tokens_weight.scalar_type() == at::kFloat) {
        return coda_qwen_forward_template<float>(*this, input_ids, cos, sin, k_cache, v_cache, cache_index);
    } else if (embed_tokens_weight.scalar_type() == at::kBFloat16) {
        return coda_qwen_forward_template<c10::BFloat16>(*this, input_ids, cos, sin, k_cache, v_cache, cache_index);
    } else {
        TORCH_CHECK(false, "CodaQwenModel only supports float32 and bfloat16 dtypes");
    }
#endif
}

}  // namespace coda::cpu
