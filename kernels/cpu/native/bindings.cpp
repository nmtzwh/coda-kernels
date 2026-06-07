#include <torch/extension.h>

#include "coda_post_ops.h"

#include <pybind11/pybind11.h>

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
}
