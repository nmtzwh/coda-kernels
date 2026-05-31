#include "coda_post_ops.h"

#include <ATen/Parallel.h>

#include <algorithm>
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

int64_t choose_m_tile(int64_t remaining) {
    return remaining >= 32 ? 32 : std::min<int64_t>(remaining, 16);
}

int64_t choose_n_tile(int64_t remaining) {
    if (remaining >= 64) return 64;
    if (remaining >= 32) return 32;
    return std::min<int64_t>(remaining, 16);
}

int64_t choose_k_tile(int64_t remaining) {
    return remaining >= 128 ? 128 : std::min<int64_t>(remaining, 64);
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
std::unordered_map<KernelKey, libxsmm_smmfunction, KernelKeyHash> kernel_cache;

libxsmm_smmfunction dispatch_smm_kernel(const KernelKey &key) {
    std::call_once(libxsmm_init_once, []() { libxsmm_init(); });

    std::lock_guard<std::mutex> guard(kernel_cache_mutex);
    auto it = kernel_cache.find(key);
    if (it != kernel_cache.end()) {
        return it->second;
    }

    const float alpha = 1.0f;
    const float beta = key.beta ? 1.0f : 0.0f;
    const libxsmm_blasint m = static_cast<libxsmm_blasint>(key.m);
    const libxsmm_blasint n = static_cast<libxsmm_blasint>(key.n);
    const libxsmm_blasint k = static_cast<libxsmm_blasint>(key.k);
    const libxsmm_blasint lda = static_cast<libxsmm_blasint>(key.lda);
    const libxsmm_blasint ldb = static_cast<libxsmm_blasint>(key.ldb);
    const libxsmm_blasint ldc = static_cast<libxsmm_blasint>(key.ldc);
    const int flags = LIBXSMM_GEMM_FLAG_NONE;
    const int prefetch = LIBXSMM_PREFETCH_NONE;

    auto fn = libxsmm_smmdispatch(
            m,
            n,
            k,
            &lda,
            &ldb,
            &ldc,
            &alpha,
            &beta,
            &flags,
            &prefetch);
    kernel_cache.emplace(key, fn);
    return fn;
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

    auto C = at::empty({M, N}, A.options().dtype(at::kFloat));
    const auto m_tiles = build_tiles(M, choose_m_tile);
    const auto n_tiles = build_tiles(N, choose_n_tile);
    const int64_t total_tiles =
            static_cast<int64_t>(m_tiles.size() * n_tiles.size());

    const float *a_base = A.data_ptr<float>();
    const float *b_base = B.data_ptr<float>();
    float *c_base = C.data_ptr<float>();

    at::parallel_for(0, total_tiles, 1, [&](int64_t begin, int64_t end) {
        for (int64_t tile = begin; tile < end; ++tile) {
            const size_t m_index = static_cast<size_t>(tile / n_tiles.size());
            const size_t n_index = static_cast<size_t>(tile % n_tiles.size());
            const int64_t m0 = m_tiles[m_index];
            const int64_t n0 = n_tiles[n_index];
            const int64_t mt = std::min<int64_t>(choose_m_tile(M - m0), M - m0);
            const int64_t nt = std::min<int64_t>(choose_n_tile(N - n0), N - n0);

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
                fn(b_tile, a_tile, c_tile);
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
