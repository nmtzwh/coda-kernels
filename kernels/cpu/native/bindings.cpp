#include <torch/extension.h>

#include "coda_post_ops.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("has_onednn", &coda::cpu::has_onednn);
    m.def("has_libxsmm", &coda::cpu::has_libxsmm);
    m.def("has_aten_vec", &coda::cpu::has_aten_vec);
    m.def("aten_vec_isa", &coda::cpu::aten_vec_isa);
    m.def(
            "execute_brgemm_postops",
            &coda::cpu::execute_brgemm_postops,
            py::arg("program_name"),
            py::arg("nodes"),
            py::arg("A"),
            py::arg("B"),
            py::arg("tensors"));
    m.def(
            "execute_libxsmm_postops",
            &coda::cpu::execute_libxsmm_postops,
            py::arg("program_name"),
            py::arg("nodes"),
            py::arg("A"),
            py::arg("B"),
            py::arg("tensors"));
    m.def(
            "execute_aten_vec_postops",
            &coda::cpu::execute_aten_vec_postops,
            py::arg("program_name"),
            py::arg("nodes"),
            py::arg("A"),
            py::arg("B"),
            py::arg("tensors"));
    m.def(
            "prepack_weight",
            &coda::cpu::prepack_weight,
            py::arg("B"));

    py::class_<coda::cpu::CodaQwenModel>(m, "CodaQwenModel")
        .def(py::init<
            at::Tensor,
            std::vector<at::Tensor>,
            std::vector<at::Tensor>,
            std::vector<at::Tensor>,
            std::vector<at::Tensor>,
            std::vector<at::Tensor>,
            std::vector<at::Tensor>,
            std::vector<at::Tensor>,
            std::vector<at::Tensor>,
            std::vector<at::Tensor>,
            at::Tensor,
            at::Tensor,
            double,
            int64_t,
            int64_t,
            int64_t,
            bool
        >(),
        py::arg("embed_tokens_weight"),
        py::arg("input_layernorm_weights"),
        py::arg("post_attention_layernorm_weights"),
        py::arg("w3_weights"),
        py::arg("qkv_biases"),
        py::arg("w0_weights"),
        py::arg("w1_weights"),
        py::arg("w2_weights"),
        py::arg("q_norm_weights"),
        py::arg("k_norm_weights"),
        py::arg("final_norm_weight"),
        py::arg("lm_head_weight"),
        py::arg("rms_norm_eps"),
        py::arg("num_heads"),
        py::arg("num_kv_heads"),
        py::arg("head_dim"),
        py::arg("is_qwen3")
        )
        .def("forward", &coda::cpu::CodaQwenModel::forward,
            py::arg("input_ids"),
            py::arg("cos"),
            py::arg("sin"),
            py::arg("k_cache"),
            py::arg("v_cache"),
            py::arg("cache_index")
        );
}

