#include "coda_post_ops.h"

#include <ATen/Parallel.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(CODA_CPU_WITH_ONEDNN)
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_ukernel.hpp>
#endif

namespace py = pybind11;

namespace coda::cpu {

bool has_onednn() {
#if defined(CODA_CPU_WITH_ONEDNN)
    return true;
#else
    return false;
#endif
}

namespace {

at::Tensor fallback_accumulate(const at::Tensor &A, const at::Tensor &B) {
    TORCH_CHECK(A.device().is_cpu(), "A must be a CPU tensor");
    TORCH_CHECK(B.device().is_cpu(), "B must be a CPU tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "BRGeMM provider expects 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "incompatible GEMM shapes");
    return at::mm(A.to(at::kFloat), B.to(at::kFloat));
}

#if defined(CODA_CPU_WITH_ONEDNN)

int64_t env_i64(const char *name, int64_t fallback) {
    const char *value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end != value && parsed > 0 ? static_cast<int64_t>(parsed) : fallback;
}

int64_t choose_m_tile(int64_t remaining) {
    const int64_t requested = env_i64("CODA_ONEDNN_M_TILE", 128);
    return std::min<int64_t>(remaining, requested);
}

int64_t choose_n_tile(int64_t remaining) {
    const int64_t requested = env_i64("CODA_ONEDNN_N_TILE", 32);
    return std::min<int64_t>(remaining, requested);
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

uint8_t *align_64(std::vector<uint8_t> &buffer) {
    const uintptr_t raw = reinterpret_cast<uintptr_t>(buffer.data());
    const uintptr_t aligned = (raw + 63U) & ~uintptr_t{63U};
    return reinterpret_cast<uint8_t *>(aligned);
}

struct BrgemmKey {
    int m;
    int n;
    int k;
    int batch_size;
    int lda;
    int ldb;
    int ldc;

    bool operator==(const BrgemmKey &other) const {
        return std::tie(m, n, k, batch_size, lda, ldb, ldc) ==
                std::tie(other.m, other.n, other.k, other.batch_size, other.lda, other.ldb, other.ldc);
    }
};

struct BrgemmKeyHash {
    size_t operator()(const BrgemmKey &key) const {
        uint64_t seed = 0;
        seed = hash_combine(seed, static_cast<uint64_t>(key.m));
        seed = hash_combine(seed, static_cast<uint64_t>(key.n));
        seed = hash_combine(seed, static_cast<uint64_t>(key.k));
        seed = hash_combine(seed, static_cast<uint64_t>(key.batch_size));
        seed = hash_combine(seed, static_cast<uint64_t>(key.lda));
        seed = hash_combine(seed, static_cast<uint64_t>(key.ldb));
        seed = hash_combine(seed, static_cast<uint64_t>(key.ldc));
        return static_cast<size_t>(seed);
    }
};

struct CachedBrgemm {
    std::shared_ptr<dnnl::ukernel::brgemm> brg;
    size_t scratchpad_size;
};

std::mutex brgemm_cache_mutex;
std::unordered_map<BrgemmKey, std::shared_ptr<CachedBrgemm>, BrgemmKeyHash> brgemm_cache;

std::shared_ptr<CachedBrgemm> dispatch_brgemm_kernel(const BrgemmKey &key) {
    using namespace dnnl;
    using namespace dnnl::ukernel;

    std::lock_guard<std::mutex> guard(brgemm_cache_mutex);
    auto it = brgemm_cache.find(key);
    if (it != brgemm_cache.end()) {
        return it->second;
    }

    auto brg = std::make_shared<brgemm>(
            static_cast<memory::dim>(key.m),
            static_cast<memory::dim>(key.n),
            static_cast<memory::dim>(key.k),
            static_cast<memory::dim>(key.batch_size),
            static_cast<memory::dim>(key.lda),
            static_cast<memory::dim>(key.ldb),
            static_cast<memory::dim>(key.ldc),
            memory::data_type::f32,
            memory::data_type::f32,
            memory::data_type::f32,
            /* allow_empty = */ true);
    if (!*brg) {
        return nullptr;
    }
    brg->set_add_C(false);
    if (!brg->finalize()) {
        return nullptr;
    }
    brg->generate();
    brg->set_hw_context();

    auto cached = std::make_shared<CachedBrgemm>(
            CachedBrgemm{brg, static_cast<size_t>(brg->get_scratchpad_size())});
    brgemm_cache.emplace(key, cached);
    return cached;
}

at::Tensor onednn_brgemm_accumulate(const at::Tensor &A, const at::Tensor &B) {
    using namespace dnnl;
    using namespace dnnl::ukernel;

    // oneDNN ukernel support is currently restricted to contiguous f32 tensors.
    // The CODA post-op chain remains independent of this mainloop.
    const bool supported_dtype =
            A.scalar_type() == at::kFloat && B.scalar_type() == at::kFloat;
    if (!supported_dtype || !A.is_contiguous() || !B.is_contiguous()) {
        return fallback_accumulate(A, B);
    }

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);
    const int64_t lda = K;
    const int64_t ldb = N;
    const int64_t ldc = N;
    const int64_t requested_k_tile = env_i64("CODA_ONEDNN_K_TILE", 256);
    const int64_t k_tile = K % requested_k_tile == 0 ? requested_k_tile : K;
    const int64_t batch_size = K / k_tile;
    const memory::data_type a_dt = memory::data_type::f32;
    const memory::data_type b_dt = memory::data_type::f32;

    const auto pack = brgemm::get_B_pack_type(a_dt, b_dt);
    if (pack == pack_type::undef || pack != pack_type::no_trans) {
        return fallback_accumulate(A, B);
    }

    auto C = at::empty({A.size(0), B.size(1)}, A.options().dtype(at::kFloat));
    const auto m_tiles = build_tiles(M, choose_m_tile);
    const auto n_tiles = build_tiles(N, choose_n_tile);
    const int64_t total_tiles =
            static_cast<int64_t>(m_tiles.size() * n_tiles.size());
    const float *a_base = A.data_ptr<float>();
    const float *b_base = B.data_ptr<float>();
    float *c_base = C.data_ptr<float>();

    at::parallel_for(0, total_tiles, 1, [&](int64_t begin, int64_t end) {
        std::vector<std::pair<memory::dim, memory::dim>> offsets;
        std::vector<uint8_t> scratchpad;
        for (int64_t tile = begin; tile < end; ++tile) {
            const size_t m_index = static_cast<size_t>(tile / n_tiles.size());
            const size_t n_index = static_cast<size_t>(tile % n_tiles.size());
            const int64_t m0 = m_tiles[m_index];
            const int64_t n0 = n_tiles[n_index];
            const int64_t mt = choose_m_tile(M - m0);
            const int64_t nt = choose_n_tile(N - n0);
            const BrgemmKey key{
                    static_cast<int>(mt),
                    static_cast<int>(nt),
                    static_cast<int>(k_tile),
                    static_cast<int>(batch_size),
                    static_cast<int>(lda),
                    static_cast<int>(ldb),
                    static_cast<int>(ldc),
            };
            auto cached = dispatch_brgemm_kernel(key);
            if (!cached) {
                throw std::runtime_error("oneDNN failed to dispatch an f32 BRGEMM ukernel");
            }
            if (scratchpad.size() < cached->scratchpad_size + 64) {
                scratchpad.resize(cached->scratchpad_size + 64);
            }
            if (static_cast<int64_t>(offsets.size()) != batch_size) {
                offsets.resize(static_cast<size_t>(batch_size));
                for (int64_t k = 0; k < batch_size; ++k) {
                    offsets[static_cast<size_t>(k)] = std::make_pair(
                            static_cast<memory::dim>(k * k_tile * static_cast<int64_t>(sizeof(float))),
                            static_cast<memory::dim>(k * k_tile * N * static_cast<int64_t>(sizeof(float))));
                }
            }
            cached->brg->execute(
                    reinterpret_cast<const uint8_t *>(a_base + m0 * K),
                    reinterpret_cast<const uint8_t *>(b_base + n0),
                    offsets,
                    c_base + m0 * N + n0,
                    align_64(scratchpad));
        }
    });
    return C;
}
#endif

}  // namespace

std::pair<at::Tensor, py::dict> execute_brgemm_postops(
        const std::string &program_name,
        const py::list &nodes,
        const at::Tensor &A,
        const at::Tensor &B,
        const py::dict &tensors) {
    (void)program_name;
    const auto parsed_nodes = parse_post_ops(nodes);
    const auto tensor_map = parse_tensor_map(tensors);

#if defined(CODA_CPU_WITH_ONEDNN)
    auto acc = onednn_brgemm_accumulate(A, B);
#else
    auto acc = fallback_accumulate(A, B);
#endif
    return run_post_ops(acc, parsed_nodes, tensor_map, A.scalar_type());
}

}  // namespace coda::cpu
