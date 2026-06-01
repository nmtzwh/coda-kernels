#include "coda_post_ops.h"

#include <ATen/Parallel.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(CODA_CPU_WITH_LIBXSMM)
#include <libxsmm.h>
#endif

namespace py = pybind11;

namespace coda::cpu {
namespace {

at::Tensor fallback_accumulate(const at::Tensor &A, const at::Tensor &B) {
    TORCH_CHECK(A.device().is_cpu(), "A must be a CPU tensor");
    TORCH_CHECK(B.device().is_cpu(), "B must be a CPU tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "LIBXSMM provider expects 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "incompatible GEMM shapes");
    return at::mm(A.to(at::kFloat), B.to(at::kFloat));
}

int64_t env_i64(const char *name, int64_t fallback);

int64_t choose_m_tile(int64_t remaining) {
    const int64_t requested = env_i64("CODA_LIBXSMM_M_TILE", 128);
    return std::min<int64_t>(remaining, requested);
}

int64_t choose_n_tile(int64_t remaining) {
    const int64_t requested = env_i64("CODA_LIBXSMM_N_TILE", 32);
    return std::min<int64_t>(remaining, requested);
}

int64_t choose_k_tile(int64_t remaining) {
    return remaining;
}

int64_t env_i64(const char *name, int64_t fallback) {
    const char *value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end != value && parsed > 0 ? static_cast<int64_t>(parsed) : fallback;
}

bool env_enabled(const char *name, bool fallback) {
    const char *value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    return value[0] != '0';
}

int64_t brgemm_k_tile(int64_t K) {
    const int64_t requested = env_i64("CODA_LIBXSMM_BR_K_TILE", 512);
    if (requested <= 0 || K % requested != 0) {
        return 0;
    }
    return requested;
}

bool use_dense_sgemm(int64_t M, int64_t K, int64_t N) {
    if (!env_enabled("CODA_LIBXSMM_DENSE_SGEMM", false)) {
        return false;
    }
    const int64_t min_flops = env_i64("CODA_LIBXSMM_DENSE_MIN_FLOPS", 32LL * 1024LL * 1024LL);
    return 2LL * M * K * N >= min_flops;
}

bool use_dense_omp() {
    return env_enabled("CODA_LIBXSMM_DENSE_OMP", true);
}

std::vector<int64_t> build_tiles(int64_t extent, int64_t (*choose_tile)(int64_t)) {
    std::vector<int64_t> starts;
    starts.reserve(static_cast<size_t>((extent + 15) / 16));
    for (int64_t offset = 0; offset < extent;) {
        starts.push_back(offset);
        offset += choose_tile(extent - offset);
    }
    return starts;
}

uint64_t hash_combine(uint64_t seed, uint64_t value) {
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

uint64_t hash_post_op_signature(
        const std::string &program_name,
        const std::vector<PostOpNode> &nodes) {
    uint64_t seed = std::hash<std::string>{}(program_name);
    for (const auto &node : nodes) {
        seed = hash_combine(seed, static_cast<uint64_t>(node.kind));
        seed = hash_combine(seed, std::hash<std::string>{}(node.output));
        seed = hash_combine(seed, std::hash<std::string>{}(node.tensor));
        seed = hash_combine(seed, static_cast<uint64_t>(node.block_size));
        seed = hash_combine(seed, std::hash<double>{}(node.value));
        seed = hash_combine(seed, static_cast<uint64_t>(node.backward));
    }
    return seed;
}

#if defined(CODA_CPU_WITH_LIBXSMM)

struct KernelKey {
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    int dtype;
    int beta;
    int use_brgemm;
    uint64_t post_op_signature;

    bool operator==(const KernelKey &other) const {
        return std::tie(m, n, k, lda, ldb, ldc, dtype, beta, use_brgemm, post_op_signature) ==
                std::tie(
                        other.m,
                        other.n,
                        other.k,
                        other.lda,
                        other.ldb,
                        other.ldc,
                        other.dtype,
                        other.beta,
                        other.use_brgemm,
                        other.post_op_signature);
    }
};

struct KernelKeyHash {
    size_t operator()(const KernelKey &key) const {
        uint64_t seed = 0;
        seed = hash_combine(seed, static_cast<uint64_t>(key.m));
        seed = hash_combine(seed, static_cast<uint64_t>(key.n));
        seed = hash_combine(seed, static_cast<uint64_t>(key.k));
        seed = hash_combine(seed, static_cast<uint64_t>(key.lda));
        seed = hash_combine(seed, static_cast<uint64_t>(key.ldb));
        seed = hash_combine(seed, static_cast<uint64_t>(key.ldc));
        seed = hash_combine(seed, static_cast<uint64_t>(key.dtype));
        seed = hash_combine(seed, static_cast<uint64_t>(key.beta));
        seed = hash_combine(seed, static_cast<uint64_t>(key.use_brgemm));
        seed = hash_combine(seed, key.post_op_signature);
        return static_cast<size_t>(seed);
    }
};

std::once_flag libxsmm_init_once;
std::mutex kernel_cache_mutex;
std::unordered_map<KernelKey, libxsmm_gemmfunction, KernelKeyHash> kernel_cache;

struct BrgemmKey {
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    int stride_a;
    int stride_b;
    int beta;
    int br_count;
    uint64_t post_op_signature;

    bool operator==(const BrgemmKey &other) const {
        return std::tie(m, n, k, lda, ldb, ldc, stride_a, stride_b, beta, br_count, post_op_signature) ==
                std::tie(
                        other.m,
                        other.n,
                        other.k,
                        other.lda,
                        other.ldb,
                        other.ldc,
                        other.stride_a,
                        other.stride_b,
                        other.beta,
                        other.br_count,
                        other.post_op_signature);
    }
};

struct BrgemmKeyHash {
    size_t operator()(const BrgemmKey &key) const {
        uint64_t seed = 0;
        seed = hash_combine(seed, static_cast<uint64_t>(key.m));
        seed = hash_combine(seed, static_cast<uint64_t>(key.n));
        seed = hash_combine(seed, static_cast<uint64_t>(key.k));
        seed = hash_combine(seed, static_cast<uint64_t>(key.lda));
        seed = hash_combine(seed, static_cast<uint64_t>(key.ldb));
        seed = hash_combine(seed, static_cast<uint64_t>(key.ldc));
        seed = hash_combine(seed, static_cast<uint64_t>(key.stride_a));
        seed = hash_combine(seed, static_cast<uint64_t>(key.stride_b));
        seed = hash_combine(seed, static_cast<uint64_t>(key.beta));
        seed = hash_combine(seed, static_cast<uint64_t>(key.br_count));
        seed = hash_combine(seed, key.post_op_signature);
        return static_cast<size_t>(seed);
    }
};

std::mutex brgemm_kernel_cache_mutex;
std::unordered_map<BrgemmKey, libxsmm_gemmfunction, BrgemmKeyHash> brgemm_kernel_cache;

libxsmm_gemmfunction dispatch_smm_kernel(const KernelKey &key) {
    std::call_once(libxsmm_init_once, []() { libxsmm_init(); });

    std::lock_guard<std::mutex> guard(kernel_cache_mutex);
    auto it = kernel_cache.find(key);
    if (it != kernel_cache.end()) {
        return it->second;
    }

    const libxsmm_blasint m = static_cast<libxsmm_blasint>(key.m);
    const libxsmm_blasint n = static_cast<libxsmm_blasint>(key.n);
    const libxsmm_blasint k = static_cast<libxsmm_blasint>(key.k);
    const libxsmm_blasint lda = static_cast<libxsmm_blasint>(key.lda);
    const libxsmm_blasint ldb = static_cast<libxsmm_blasint>(key.ldb);
    const libxsmm_blasint ldc = static_cast<libxsmm_blasint>(key.ldc);
    const auto shape = libxsmm_create_gemm_shape(
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32);
    const auto flags = static_cast<libxsmm_bitfield>(
            key.beta ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_BETA_0);
    auto fn = libxsmm_dispatch_gemm(shape, flags, LIBXSMM_GEMM_PREFETCH_NONE);
    kernel_cache.emplace(key, fn);
    return fn;
}

libxsmm_gemmfunction dispatch_brgemm_kernel(const BrgemmKey &key) {
    std::call_once(libxsmm_init_once, []() { libxsmm_init(); });

    std::lock_guard<std::mutex> guard(brgemm_kernel_cache_mutex);
    auto it = brgemm_kernel_cache.find(key);
    if (it != brgemm_kernel_cache.end()) {
        return it->second;
    }

    const libxsmm_blasint m = static_cast<libxsmm_blasint>(key.m);
    const libxsmm_blasint n = static_cast<libxsmm_blasint>(key.n);
    const libxsmm_blasint k = static_cast<libxsmm_blasint>(key.k);
    const libxsmm_blasint lda = static_cast<libxsmm_blasint>(key.lda);
    const libxsmm_blasint ldb = static_cast<libxsmm_blasint>(key.ldb);
    const libxsmm_blasint ldc = static_cast<libxsmm_blasint>(key.ldc);
    const libxsmm_blasint stride_a = static_cast<libxsmm_blasint>(key.stride_a);
    const libxsmm_blasint stride_b = static_cast<libxsmm_blasint>(key.stride_b);
    const auto shape = libxsmm_create_gemm_shape(
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32);
    const auto flags = static_cast<libxsmm_bitfield>(
            key.beta ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_BETA_0);
    const auto br_config = libxsmm_create_gemm_batch_reduce_config(
            LIBXSMM_GEMM_BATCH_REDUCE_STRIDE,
            stride_a,
            stride_b,
            static_cast<unsigned char>(std::min(key.br_count, 255)));
    auto fn = libxsmm_dispatch_brgemm(shape, flags, LIBXSMM_GEMM_PREFETCH_NONE, br_config);
    brgemm_kernel_cache.emplace(key, fn);
    return fn;
}

at::Tensor libxsmm_dense_accumulate(
        const at::Tensor &A,
        const at::Tensor &B) {
    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);
    auto C = at::empty({M, N}, A.options().dtype(at::kFloat));

    const char trans = 'N';
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const libxsmm_blasint cm = static_cast<libxsmm_blasint>(N);
    const libxsmm_blasint cn = static_cast<libxsmm_blasint>(M);
    const libxsmm_blasint ck = static_cast<libxsmm_blasint>(K);
    const libxsmm_blasint lda = static_cast<libxsmm_blasint>(N);
    const libxsmm_blasint ldb = static_cast<libxsmm_blasint>(K);
    const libxsmm_blasint ldc = static_cast<libxsmm_blasint>(N);

    if (use_dense_omp()) {
        const float *b_base = B.data_ptr<float>();
        const float *a_base = A.data_ptr<float>();
        float *c_base = C.data_ptr<float>();
        at::parallel_for(0, M, 16, [&](int64_t begin, int64_t end) {
            const libxsmm_blasint block_rows = static_cast<libxsmm_blasint>(end - begin);
            const float *a_block = a_base + begin * K;
            float *c_block = c_base + begin * N;
            libxsmm_sgemm(
                    &trans,
                    &trans,
                    &cm,
                    &block_rows,
                    &ck,
                    &alpha,
                    b_base,
                    &lda,
                    a_block,
                    &ldb,
                    &beta,
                    c_block,
                    &ldc);
        });
    } else {
        libxsmm_sgemm(
                &trans,
                &trans,
                &cm,
                &cn,
                &ck,
                &alpha,
                B.data_ptr<float>(),
                &lda,
                A.data_ptr<float>(),
                &ldb,
                &beta,
                C.data_ptr<float>(),
                &ldc);
    }
    return C;
}

at::Tensor libxsmm_accumulate(
        const at::Tensor &A,
        const at::Tensor &B,
        uint64_t post_op_signature) {
    TORCH_CHECK(A.device().is_cpu(), "A must be a CPU tensor");
    TORCH_CHECK(B.device().is_cpu(), "B must be a CPU tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "LIBXSMM provider expects 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "incompatible GEMM shapes");

    if (A.scalar_type() != at::kFloat || B.scalar_type() != at::kFloat ||
            !A.is_contiguous() || !B.is_contiguous()) {
        return fallback_accumulate(A, B);
    }

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);
    if (M == 0 || N == 0) {
        return at::empty({M, N}, A.options().dtype(at::kFloat));
    }
    if (K == 0) {
        return at::zeros({M, N}, A.options().dtype(at::kFloat));
    }

    if (use_dense_sgemm(M, K, N)) {
        return libxsmm_dense_accumulate(A, B);
    }

    auto C = at::empty({M, N}, A.options().dtype(at::kFloat));
    const auto m_tiles = build_tiles(M, choose_m_tile);
    const auto n_tiles = build_tiles(N, choose_n_tile);
    const int64_t total_tiles =
            static_cast<int64_t>(m_tiles.size() * n_tiles.size());

    const float *a_base = A.data_ptr<float>();
    const float *b_base = B.data_ptr<float>();
    float *c_base = C.data_ptr<float>();
    const int64_t br_k_tile = brgemm_k_tile(K);
    const bool use_brgemm =
            env_enabled("CODA_LIBXSMM_USE_BRGEMM", true) &&
            br_k_tile > 0 &&
            K >= br_k_tile * 2;

    at::parallel_for(0, total_tiles, 1, [&](int64_t begin, int64_t end) {
        for (int64_t tile = begin; tile < end; ++tile) {
            const size_t m_index = static_cast<size_t>(tile / n_tiles.size());
            const size_t n_index = static_cast<size_t>(tile % n_tiles.size());
            const int64_t m0 = m_tiles[m_index];
            const int64_t n0 = n_tiles[n_index];
            const int64_t mt = std::min<int64_t>(choose_m_tile(M - m0), M - m0);
            const int64_t nt = std::min<int64_t>(choose_n_tile(N - n0), N - n0);

            if (use_brgemm) {
                const int64_t br_count = K / br_k_tile;
                const BrgemmKey key{
                        static_cast<int>(nt),
                        static_cast<int>(mt),
                        static_cast<int>(br_k_tile),
                        static_cast<int>(N),
                        static_cast<int>(K),
                        static_cast<int>(N),
                        static_cast<int>(br_k_tile * N * static_cast<int64_t>(sizeof(float))),
                        static_cast<int>(br_k_tile * static_cast<int64_t>(sizeof(float))),
                        0,
                        static_cast<int>(br_count),
                        post_op_signature,
                };
                auto fn = dispatch_brgemm_kernel(key);
                if (fn == nullptr) {
                    throw std::runtime_error("LIBXSMM failed to dispatch an fp32 strided BRGEMM kernel");
                }

                const float *b_tile = b_base + n0;
                const float *a_tile = a_base + m0 * K;
                float *c_tile = c_base + m0 * N + n0;
                const unsigned long long count = static_cast<unsigned long long>(br_count);
                libxsmm_gemm_param args{};
                args.op.tertiary = const_cast<unsigned long long *>(&count);
                args.a.primary = const_cast<float *>(b_tile);
                args.b.primary = const_cast<float *>(a_tile);
                args.c.primary = c_tile;
                fn(&args);
                continue;
            }

            for (int64_t k0 = 0; k0 < K;) {
                const int64_t kt = std::min<int64_t>(choose_k_tile(K - k0), K - k0);
                const KernelKey key{
                        static_cast<int>(nt),
                        static_cast<int>(mt),
                        static_cast<int>(kt),
                        static_cast<int>(N),
                        static_cast<int>(K),
                        static_cast<int>(N),
                        static_cast<int>(at::kFloat),
                        k0 == 0 ? 0 : 1,
                        0,
                        post_op_signature,
                };
                auto fn = dispatch_smm_kernel(key);
                if (fn == nullptr) {
                    throw std::runtime_error("LIBXSMM failed to dispatch an fp32 SMM kernel");
                }

                // Row-major C = A x B is evaluated as column-major C^T = B^T x A^T.
                const float *b_tile = b_base + k0 * N + n0;
                const float *a_tile = a_base + m0 * K + k0;
                float *c_tile = c_base + m0 * N + n0;
                libxsmm_gemm_param args{};
                args.a.primary = const_cast<float *>(b_tile);
                args.b.primary = const_cast<float *>(a_tile);
                args.c.primary = c_tile;
                fn(&args);
                k0 += kt;
            }
        }
    });

    return C;
}

#endif  // defined(CODA_CPU_WITH_LIBXSMM)

}  // namespace

bool has_libxsmm() {
#if defined(CODA_CPU_WITH_LIBXSMM)
    std::call_once(libxsmm_init_once, []() { libxsmm_init(); });
    return true;
#else
    return false;
#endif
}

std::pair<at::Tensor, py::dict> execute_libxsmm_postops(
        const std::string &program_name,
        const py::list &nodes,
        const at::Tensor &A,
        const at::Tensor &B,
        const py::dict &tensors) {
    const auto parsed_nodes = parse_post_ops(nodes);
    const auto tensor_map = parse_tensor_map(tensors);

#if defined(CODA_CPU_WITH_LIBXSMM)
    auto acc = libxsmm_accumulate(A, B, hash_post_op_signature(program_name, parsed_nodes));
#else
    (void)program_name;
    auto acc = fallback_accumulate(A, B);
#endif
    return run_post_ops(acc, parsed_nodes, tensor_map, A.scalar_type());
}

}  // namespace coda::cpu
