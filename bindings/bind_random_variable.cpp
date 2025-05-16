#include <pybind11/pybind11.h>
#include "shrew/random_variable.hpp"
#include "shrew/normal_distribution.hpp"
#include "shrew/compound_distribution.hpp"

namespace py = pybind11;
using namespace shrew::random_variable;

void bind_random_variable(py::module_ &m) {
    py::class_<RandomVariable<NormalDistribution>>(m, "RandomVariable")
        .def(py::init<NormalDistribution>(), py::arg("NormalDistribution"))
        .def_readonly("pdist", &RandomVariable<NormalDistribution>::probability_distribution)
        .def("__add__", [](const RandomVariable<NormalDistribution>& lhs, const RandomVariable<NormalDistribution>& rhs) {
            return lhs + rhs;
        })
        .def("__radd__", [](const RandomVariable<NormalDistribution>& rhs, double lhs) {
            return lhs + rhs;
        })
        .def("__add__", [](const RandomVariable<NormalDistribution>& lhs, double rhs) {
            return lhs + rhs;
        })
        .def("__sub__", [](const RandomVariable<NormalDistribution>& lhs, const RandomVariable<NormalDistribution>& rhs) {
            return lhs - rhs;
        })
        .def("__rsub__", [](const RandomVariable<NormalDistribution>& rhs, double lhs) {
            return lhs - rhs;
        })
        .def("__sub__", [](const RandomVariable<NormalDistribution>& lhs, double rhs) {
            return lhs - rhs;
        })
        .def("__mul__", [](const RandomVariable<NormalDistribution>& lhs, const RandomVariable<NormalDistribution>& rhs) {
            return lhs * rhs;
        })
        .def("__rmul__", [](const RandomVariable<NormalDistribution>& rhs, double lhs) {
            return lhs * rhs;
        })
        .def("__mul__", [](const RandomVariable<NormalDistribution>& lhs, double rhs) {
            return lhs * rhs;
        })
        .def("__truediv__", [](const RandomVariable<NormalDistribution>& lhs, const RandomVariable<NormalDistribution>& rhs) {
            return lhs / rhs;
        })
        .def("__rtruediv__", [](const RandomVariable<NormalDistribution>& rhs, double lhs) {
            return lhs / rhs;
        })
        .def("__truediv__", [](const RandomVariable<NormalDistribution>& lhs, double rhs) {
            return lhs / rhs;
        })
        .def("__pow__", [](const RandomVariable<NormalDistribution>& lhs, const RandomVariable<NormalDistribution>& rhs) {
            return lhs ^ rhs;
        })
        .def("__rpow__", [](const RandomVariable<NormalDistribution>& rhs, double lhs) {
            return lhs ^ rhs;
        })
        .def("__pow__", [](const RandomVariable<NormalDistribution>& lhs, double rhs) {
            return lhs ^ rhs;
        });

    py::class_<RandomVariable<CompoundDistribution<NormalDistribution, NormalDistribution>>>(m, "CompoundRandomVariable")
        .def_readonly("pdist", &RandomVariable<CompoundDistribution<NormalDistribution, NormalDistribution>>::probability_distribution);
    py::class_<RandomVariable<CompoundDistribution<double, NormalDistribution>>>(m, "CompoundRandomVariable_dn")
        .def_readonly("pdist", &RandomVariable<CompoundDistribution<double, NormalDistribution>>::probability_distribution);
    py::class_<RandomVariable<CompoundDistribution<NormalDistribution, double>>>(m, "CompoundRandomVariable_nd")
        .def_readonly("pdist", &RandomVariable<CompoundDistribution<NormalDistribution, double>>::probability_distribution);
}
