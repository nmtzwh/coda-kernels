#include "coda_post_ops.h"

#include <stdexcept>
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

    const auto M = static_cast<memory::dim>(A.size(0));
    const auto K = static_cast<memory::dim>(A.size(1));
    const auto N = static_cast<memory::dim>(B.size(1));
    const memory::dim lda = K;
    const memory::dim ldb = N;
    const memory::dim ldc = N;
    const memory::data_type a_dt = memory::data_type::f32;
    const memory::data_type b_dt = memory::data_type::f32;
    const memory::data_type c_dt = memory::data_type::f32;

    const auto pack = brgemm::get_B_pack_type(a_dt, b_dt);
    if (pack == pack_type::undef || pack != pack_type::no_trans) {
        return fallback_accumulate(A, B);
    }

    brgemm brg(
            M,
            N,
            K,
            /* batch_size = */ 1,
            lda,
            ldb,
            ldc,
            a_dt,
            b_dt,
            c_dt,
            /* allow_empty = */ true);
    if (!brg) {
        return fallback_accumulate(A, B);
    }

    brg.set_add_C(false);
    if (!brg.finalize()) {
        return fallback_accumulate(A, B);
    }
    brg.generate();
    brg.set_hw_context();

    auto C = at::empty({A.size(0), B.size(1)}, A.options().dtype(at::kFloat));
    std::vector<uint8_t> scratchpad(brg.get_scratchpad_size());
    std::vector<std::pair<memory::dim, memory::dim>> offsets;
    offsets.emplace_back(0, 0);
    brg.execute(
            reinterpret_cast<const uint8_t *>(A.data_ptr<float>()),
            reinterpret_cast<const uint8_t *>(B.data_ptr<float>()),
            offsets,
            C.data_ptr<float>(),
            scratchpad.data());
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
