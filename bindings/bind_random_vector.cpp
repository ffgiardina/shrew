#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include "shrew/multivariate_normal.hpp"


namespace py = pybind11;
using namespace shrew::random_vector;

void bind_random_vector(py::module_ &m) {
    py::class_<MultivariateNormal>(m, "RandomVector")
        .def(py::init<Eigen::VectorXd , Eigen::MatrixXd>(), py::arg("mu"), py::arg("K"))
        .def("joint_pdf", &MultivariateNormal::joint_pdf, py::arg("x"))
        .def_readonly("mu", &MultivariateNormal::mu)
        .def_readonly("K", &MultivariateNormal::K);

    m.def("get_marginal", &getMarginal, py::arg("random_vector"), py::arg("marginal_indices"));
    m.def("get_conditional", py::overload_cast<MultivariateNormal, Eigen::VectorXi, char, Eigen::MatrixXd>(&getConditional), py::arg("random_vector"), py::arg("conditional_indices"), py::arg("operator"), py::arg("value"));
}
