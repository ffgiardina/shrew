#include <pybind11/pybind11.h>
#include "shrew/normal_distribution.hpp"

namespace py = pybind11;
using namespace shrew::random_variable;

void bind_normal_distribution(py::module_ &m) {
    py::class_<NormalDistribution>(m, "NormalDistribution")
        .def(py::init<double, double>(), py::arg("mean"), py::arg("stddev"))
        .def("pdf", &NormalDistribution::Pdf, py::arg("x"));
}
