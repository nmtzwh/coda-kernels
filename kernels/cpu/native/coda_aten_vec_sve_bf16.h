#pragma once

#if !defined(__aarch64__) && !defined(_M_ARM64)
#error "CODA SVE BF16 kernels require an AArch64 target"
#endif

#if !defined(__ARM_FEATURE_SVE) || !defined(__ARM_FEATURE_SVE_BF16) || \
        !defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#error "CODA SVE BF16 kernels require SVE BF16 vector arithmetic"
#endif

#if !defined(__ARM_FEATURE_SVE_BITS) || __ARM_FEATURE_SVE_BITS != 256
#error "CODA aten-vec SVE BF16 kernels currently target fixed SVE256"
#endif

#include <arm_sve.h>

#include <cstdint>

namespace coda::cpu::aten_vec_sve_bf16 {

constexpr int64_t kVecSize = __ARM_FEATURE_SVE_BITS / 32;

typedef uint16_t aliased_uint16_t __attribute__((__may_alias__));
typedef uint32_t aliased_uint32_t __attribute__((__may_alias__));

template <typename BF16>
inline uint32_t load_bf16_pair_bits(const BF16 *ptr) {
    static_assert(sizeof(BF16) == sizeof(uint16_t));
    return *reinterpret_cast<const aliased_uint32_t*>(ptr);
}

template <typename BF16>
inline svbfloat16_t load_bf16_vector(const BF16 *ptr) {
    static_assert(sizeof(BF16) == sizeof(uint16_t));
    const auto *bits = reinterpret_cast<const aliased_uint16_t*>(ptr);
    return svreinterpret_bf16_u16(svld1_u16(svptrue_b16(), bits));
}

inline svbfloat16_t broadcast_bf16_pair(uint32_t pair_bits) {
    return svreinterpret_bf16_u32(svdup_n_u32(pair_bits));
}

template <bool accumulate>
inline svfloat32_t load_accumulator(const float *ptr) {
    if constexpr (accumulate) {
        return svld1_f32(svptrue_b32(), ptr);
    } else {
        (void)ptr;
        return svdup_n_f32(0.0f);
    }
}

inline void store_accumulator(float *ptr, svfloat32_t values) {
    svst1_f32(svptrue_b32(), ptr, values);
}

template <typename BF16, int64_t rows, int64_t col_vectors, bool accumulate>
inline void gemm_microkernel(
        const BF16 *a_base,
        const BF16 *b_panel,
        float *acc_base,
        int64_t K,
        int64_t block_n,
        int64_t k_begin,
        int64_t k_end) {
    static_assert(rows >= 1 && rows <= 4);
    static_assert(col_vectors >= 1 && col_vectors <= 4);

    svfloat32_t c00 = load_accumulator<accumulate>(acc_base);
    svfloat32_t c01 = svdup_n_f32(0.0f);
    svfloat32_t c02 = svdup_n_f32(0.0f);
    svfloat32_t c03 = svdup_n_f32(0.0f);
    svfloat32_t c10 = svdup_n_f32(0.0f);
    svfloat32_t c11 = svdup_n_f32(0.0f);
    svfloat32_t c12 = svdup_n_f32(0.0f);
    svfloat32_t c13 = svdup_n_f32(0.0f);
    svfloat32_t c20 = svdup_n_f32(0.0f);
    svfloat32_t c21 = svdup_n_f32(0.0f);
    svfloat32_t c22 = svdup_n_f32(0.0f);
    svfloat32_t c23 = svdup_n_f32(0.0f);
    svfloat32_t c30 = svdup_n_f32(0.0f);
    svfloat32_t c31 = svdup_n_f32(0.0f);
    svfloat32_t c32 = svdup_n_f32(0.0f);
    svfloat32_t c33 = svdup_n_f32(0.0f);

    if constexpr (col_vectors > 1) {
        c01 = load_accumulator<accumulate>(acc_base + kVecSize);
    }
    if constexpr (col_vectors > 2) {
        c02 = load_accumulator<accumulate>(acc_base + 2 * kVecSize);
    }
    if constexpr (col_vectors > 3) {
        c03 = load_accumulator<accumulate>(acc_base + 3 * kVecSize);
    }
    if constexpr (rows > 1) {
        c10 = load_accumulator<accumulate>(acc_base + block_n);
        if constexpr (col_vectors > 1) {
            c11 = load_accumulator<accumulate>(acc_base + block_n + kVecSize);
        }
        if constexpr (col_vectors > 2) {
            c12 = load_accumulator<accumulate>(acc_base + block_n + 2 * kVecSize);
        }
        if constexpr (col_vectors > 3) {
            c13 = load_accumulator<accumulate>(acc_base + block_n + 3 * kVecSize);
        }
    }
    if constexpr (rows > 2) {
        c20 = load_accumulator<accumulate>(acc_base + 2 * block_n);
        if constexpr (col_vectors > 1) {
            c21 = load_accumulator<accumulate>(acc_base + 2 * block_n + kVecSize);
        }
        if constexpr (col_vectors > 2) {
            c22 = load_accumulator<accumulate>(acc_base + 2 * block_n + 2 * kVecSize);
        }
        if constexpr (col_vectors > 3) {
            c23 = load_accumulator<accumulate>(acc_base + 2 * block_n + 3 * kVecSize);
        }
    }
    if constexpr (rows > 3) {
        c30 = load_accumulator<accumulate>(acc_base + 3 * block_n);
        if constexpr (col_vectors > 1) {
            c31 = load_accumulator<accumulate>(acc_base + 3 * block_n + kVecSize);
        }
        if constexpr (col_vectors > 2) {
            c32 = load_accumulator<accumulate>(acc_base + 3 * block_n + 2 * kVecSize);
        }
        if constexpr (col_vectors > 3) {
            c33 = load_accumulator<accumulate>(acc_base + 3 * block_n + 3 * kVecSize);
        }
    }

    const int64_t p_begin = k_begin / 2;
    const int64_t p_end = (k_end + 1) / 2;
    const BF16 *b_ptr = b_panel + p_begin * 2 * block_n;
    const BF16 *a0_ptr = a_base + 2 * p_begin;
    const BF16 *a1_ptr = a0_ptr + K;
    const BF16 *a2_ptr = a0_ptr + 2 * K;
    const BF16 *a3_ptr = a0_ptr + 3 * K;

    for (int64_t p = p_begin; p < p_end; ++p) {
        (void)p;
        const svbfloat16_t b0 = load_bf16_vector(b_ptr);
        svbfloat16_t b1;
        svbfloat16_t b2;
        svbfloat16_t b3;
        if constexpr (col_vectors > 1) {
            b1 = load_bf16_vector(b_ptr + 2 * kVecSize);
        }
        if constexpr (col_vectors > 2) {
            b2 = load_bf16_vector(b_ptr + 4 * kVecSize);
        }
        if constexpr (col_vectors > 3) {
            b3 = load_bf16_vector(b_ptr + 6 * kVecSize);
        }

        svbfloat16_t a = broadcast_bf16_pair(load_bf16_pair_bits(a0_ptr));
        c00 = svbfdot_f32(c00, a, b0);
        if constexpr (col_vectors > 1) {
            c01 = svbfdot_f32(c01, a, b1);
        }
        if constexpr (col_vectors > 2) {
            c02 = svbfdot_f32(c02, a, b2);
        }
        if constexpr (col_vectors > 3) {
            c03 = svbfdot_f32(c03, a, b3);
        }

        if constexpr (rows > 1) {
            a = broadcast_bf16_pair(load_bf16_pair_bits(a1_ptr));
            c10 = svbfdot_f32(c10, a, b0);
            if constexpr (col_vectors > 1) {
                c11 = svbfdot_f32(c11, a, b1);
            }
            if constexpr (col_vectors > 2) {
                c12 = svbfdot_f32(c12, a, b2);
            }
            if constexpr (col_vectors > 3) {
                c13 = svbfdot_f32(c13, a, b3);
            }
        }

        if constexpr (rows > 2) {
            a = broadcast_bf16_pair(load_bf16_pair_bits(a2_ptr));
            c20 = svbfdot_f32(c20, a, b0);
            if constexpr (col_vectors > 1) {
                c21 = svbfdot_f32(c21, a, b1);
            }
            if constexpr (col_vectors > 2) {
                c22 = svbfdot_f32(c22, a, b2);
            }
            if constexpr (col_vectors > 3) {
                c23 = svbfdot_f32(c23, a, b3);
            }
        }

        if constexpr (rows > 3) {
            a = broadcast_bf16_pair(load_bf16_pair_bits(a3_ptr));
            c30 = svbfdot_f32(c30, a, b0);
            if constexpr (col_vectors > 1) {
                c31 = svbfdot_f32(c31, a, b1);
            }
            if constexpr (col_vectors > 2) {
                c32 = svbfdot_f32(c32, a, b2);
            }
            if constexpr (col_vectors > 3) {
                c33 = svbfdot_f32(c33, a, b3);
            }
        }

        b_ptr += 2 * block_n;
        a0_ptr += 2;
        a1_ptr += 2;
        a2_ptr += 2;
        a3_ptr += 2;
    }

    store_accumulator(acc_base, c00);
    if constexpr (col_vectors > 1) {
        store_accumulator(acc_base + kVecSize, c01);
    }
    if constexpr (col_vectors > 2) {
        store_accumulator(acc_base + 2 * kVecSize, c02);
    }
    if constexpr (col_vectors > 3) {
        store_accumulator(acc_base + 3 * kVecSize, c03);
    }
    if constexpr (rows > 1) {
        store_accumulator(acc_base + block_n, c10);
        if constexpr (col_vectors > 1) {
            store_accumulator(acc_base + block_n + kVecSize, c11);
        }
        if constexpr (col_vectors > 2) {
            store_accumulator(acc_base + block_n + 2 * kVecSize, c12);
        }
        if constexpr (col_vectors > 3) {
            store_accumulator(acc_base + block_n + 3 * kVecSize, c13);
        }
    }
    if constexpr (rows > 2) {
        store_accumulator(acc_base + 2 * block_n, c20);
        if constexpr (col_vectors > 1) {
            store_accumulator(acc_base + 2 * block_n + kVecSize, c21);
        }
        if constexpr (col_vectors > 2) {
            store_accumulator(acc_base + 2 * block_n + 2 * kVecSize, c22);
        }
        if constexpr (col_vectors > 3) {
            store_accumulator(acc_base + 2 * block_n + 3 * kVecSize, c23);
        }
    }
    if constexpr (rows > 3) {
        store_accumulator(acc_base + 3 * block_n, c30);
        if constexpr (col_vectors > 1) {
            store_accumulator(acc_base + 3 * block_n + kVecSize, c31);
        }
        if constexpr (col_vectors > 2) {
            store_accumulator(acc_base + 3 * block_n + 2 * kVecSize, c32);
        }
        if constexpr (col_vectors > 3) {
            store_accumulator(acc_base + 3 * block_n + 3 * kVecSize, c33);
        }
    }
}

}  // namespace coda::cpu::aten_vec_sve_bf16
