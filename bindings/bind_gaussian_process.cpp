#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include "shrew/gaussian_process/gaussian_process.hpp"
#include "shrew/gaussian_process/kernel.hpp"
#include "shrew/gaussian_process/maternKernel.hpp"
#include "shrew/gaussian_process/maternKernelNoise.hpp"
#include "shrew/gaussian_process/squaredExponentialKernel.hpp"

namespace py = pybind11;
using namespace gaussian_process;
using namespace gaussian_process::kernel;

void bind_gaussian_process(py::module_ &m) {
    py::class_<GaussianProcess>(m, "GaussianProcess")
        .def(py::init<Eigen::VectorXd , Eigen::VectorXd, std::vector<int>, Kernel&>(), py::arg("x"), py::arg("y"), py::arg("conditional_indices"), py::arg("kernel"))
        .def("log_marginal_likelihood", py::overload_cast<>(&GaussianProcess::LogMarginalLikelihood))
        .def("optimize", &GaussianProcess::OptimizeHyperparameters, py::arg("progress_output") = true)
        .def("get_posterior", &GaussianProcess::GetPosteriorGP);

    py::class_<Kernel>(m, "Kernel");

    py::class_<MaternNoise, Kernel>(m, "MaternNoise")
        .def(py::init<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<int>, std::vector<int>, MaternSmoothness>(), py::arg("hyperparameters"), py::arg("lower_bounds"), py::arg("upper_bounds"), py::arg("conditional_indices"), py::arg("conditional_indices_noise2"), py::arg("matern_smoothness"))
        .def("get_hyperparameters", &MaternNoise::GetHyperparameters);

    py::class_<Matern, Kernel>(m, "Matern")
        .def(py::init<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<int>, MaternSmoothness>(), py::arg("hyperparameters"), py::arg("lower_bounds"), py::arg("upper_bounds"), py::arg("conditional_indices"), py::arg("matern_smoothness"))
        .def("get_hyperparameters", &Matern::GetHyperparameters);

    py::class_<SquaredExponential, Kernel>(m, "SquaredExponential")
        .def(py::init<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<int>>(), py::arg("hyperparameters"), py::arg("lower_bounds"), py::arg("upper_bounds"), py::arg("conditional_indices"))
        .def("get_hyperparameters", &SquaredExponential::GetHyperparameters);

    py::enum_<MaternSmoothness>(m, "MaternSmoothness")
        .value("NU_0_5", MaternSmoothness::NU_0_5)
        .value("NU_1_5", MaternSmoothness::NU_1_5)
        .value("NU_2_5", MaternSmoothness::NU_2_5)
        .export_values();
}
