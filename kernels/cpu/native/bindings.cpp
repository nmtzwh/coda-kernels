#include <torch/extension.h>

#include "coda_post_ops.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("has_onednn", &coda::cpu::has_onednn);
    m.def(
            "execute_brgemm_postops",
            &coda::cpu::execute_brgemm_postops,
            py::arg("program_name"),
            py::arg("nodes"),
            py::arg("A"),
            py::arg("B"),
            py::arg("tensors"));
}
