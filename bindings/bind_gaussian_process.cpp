#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include "shrew/gaussian_process/gaussian_process.hpp"
#include "shrew/gaussian_process/kernel.hpp"
#include "shrew/gaussian_process/matern_kernel.hpp"
#include "shrew/gaussian_process/matern_kernel_extended.hpp"
#include "shrew/gaussian_process/squared_exponential_kernel.hpp"

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

    py::class_<Hyperparameters, std::shared_ptr<Hyperparameters>>(m, "Hyperparameters");

    py::class_<SEHyperparams, Hyperparameters, std::shared_ptr<SEHyperparams>>(m, "SEHyperparams")
        .def(py::init<double, double, double>(), py::arg("lengthscale"), py::arg("noise_stdv"), py::arg("signal_stdv"));

    py::class_<MaternHyperparams, Hyperparameters, std::shared_ptr<MaternHyperparams>>(m, "MaternHyperparams")
        .def(py::init<double, double, double, MaternSmoothness>(), 
             py::arg("lengthscale"), py::arg("noise_stdv"), py::arg("signal_stdv"), py::arg("nu"));

    py::class_<MaternExtendedHyperparams, MaternHyperparams, std::shared_ptr<MaternExtendedHyperparams>>(m, "MaternExtendedHyperparams")
        .def(py::init<double, double, double, MaternSmoothness, std::vector<double>>(), 
             py::arg("lengthscale"), py::arg("noise_stdv"), py::arg("signal_stdv"), py::arg("nu"), py::arg("override_noise_stdv"));

    py::class_<MaternExtended, Kernel>(m, "MaternExtended")
        .def(py::init<std::shared_ptr<Hyperparameters>, std::shared_ptr<Hyperparameters>, std::shared_ptr<Hyperparameters>, std::vector<int>, std::vector<int>>(), py::arg("hyperparameters"), py::arg("lower_bounds"), py::arg("upper_bounds"), py::arg("conditional_indices"), py::arg("override_conditional_indices"))
        .def("get_hyperparameters", &MaternExtended::GetHyperparameters);

    py::class_<Matern, Kernel>(m, "Matern")
        .def(py::init<std::shared_ptr<Hyperparameters>, std::shared_ptr<Hyperparameters>, std::shared_ptr<Hyperparameters>, std::vector<int>>(), py::arg("hyperparameters"), py::arg("lower_bounds"), py::arg("upper_bounds"), py::arg("conditional_indices"))
        .def("get_hyperparameters", &Matern::GetHyperparameters);

    py::class_<SquaredExponential, Kernel>(m, "SquaredExponential")
        .def(py::init<std::shared_ptr<Hyperparameters>, std::shared_ptr<Hyperparameters>, std::shared_ptr<Hyperparameters>, std::vector<int>>(), py::arg("hyperparameters"), py::arg("lower_bounds"), py::arg("upper_bounds"), py::arg("conditional_indices"))
        .def("get_hyperparameters", &SquaredExponential::GetHyperparameters);

    py::enum_<MaternSmoothness>(m, "MaternSmoothness")
        .value("NU_0_5", MaternSmoothness::NU_0_5)
        .value("NU_1_5", MaternSmoothness::NU_1_5)
        .value("NU_2_5", MaternSmoothness::NU_2_5)
        .export_values();
}
