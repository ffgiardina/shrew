#include <pybind11/pybind11.h>
#include "shrew/normal_distribution.hpp"
#include "shrew/compound_distribution.hpp"

namespace py = pybind11;
using namespace shrew::random_variable;

void bind_distributions(py::module_ &m) {
    py::class_<NormalDistribution>(m, "NormalDistribution")
        .def(py::init<double, double>(), py::arg("mean"), py::arg("stddev"))
        .def("pdf", &NormalDistribution::Pdf, py::arg("x"))
        .def("cdf", &NormalDistribution::Cdf, py::arg("x"));

    py::class_<CompoundDistribution<NormalDistribution, NormalDistribution>>(m, "CompoundDistribution")
        .def("pdf", &CompoundDistribution<NormalDistribution, NormalDistribution>::Pdf, py::arg("x"))
        .def("cdf", &CompoundDistribution<NormalDistribution, NormalDistribution>::Cdf, py::arg("x"));
    py::class_<CompoundDistribution<double, NormalDistribution>>(m, "CompoundDistribution_dn")
        .def("pdf", &CompoundDistribution<double, NormalDistribution>::Pdf, py::arg("x"))
        .def("cdf", &CompoundDistribution<double, NormalDistribution>::Cdf, py::arg("x"));
    py::class_<CompoundDistribution<NormalDistribution, double>>(m, "CompoundDistribution_nd")
        .def("pdf", &CompoundDistribution<NormalDistribution, double>::Pdf, py::arg("x"))
        .def("cdf", &CompoundDistribution<NormalDistribution, double>::Cdf, py::arg("x"));
}
