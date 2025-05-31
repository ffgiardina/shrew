#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <memory>
#include "shrew/gaussian_process/gaussian_process.hpp"
#include "shrew/gaussian_process/kernel.hpp"
#include "shrew/gaussian_process/matern_kernel.hpp"
#include "shrew/gaussian_process/matern_kernel_extended.hpp"
#include "shrew/gaussian_process/squared_exponential_kernel.hpp"

namespace py = pybind11;
using namespace gaussian_process;
using namespace gaussian_process::kernel;

void bind_gaussian_process(py::module_ &m) {
    py::class_<HpOptimizationMeta>(m, "HpOptimizationMeta")
        .def(py::init<>())
        .def_readwrite("iter", &HpOptimizationMeta::iter)
        .def_readwrite("running_max", &HpOptimizationMeta::running_max)
        .def_readwrite("optim_progress_intervall", &HpOptimizationMeta::optim_progress_intervall)
        .def_readwrite("progress_output", &HpOptimizationMeta::progress_output)
        .def_readwrite("algo", &HpOptimizationMeta::algo)
        .def_readwrite("y_", &HpOptimizationMeta::y_)
        .def_readwrite("x_", &HpOptimizationMeta::x_)
        .def_readwrite("kernel_", &HpOptimizationMeta::kernel_);

    py::class_<GaussianProcess>(m, "GaussianProcess")
        .def(py::init<Eigen::VectorXd , Eigen::VectorXd, Eigen::VectorXi, Kernel&>(), py::arg("x"), py::arg("y"), py::arg("conditional_indices"), py::arg("kernel"))
        .def("log_marginal_likelihood", py::overload_cast<>(&GaussianProcess::LogMarginalLikelihood))
        .def("optimize", &GaussianProcess::OptimizeHyperparameters, py::arg("progress_output") = true)
        .def("get_posterior", &GaussianProcess::GetPosteriorGP)
        .def_readwrite("x", &GaussianProcess::x)
        .def_readwrite("y", &GaussianProcess::y)
        .def_readwrite("conditional_indices", &GaussianProcess::conditional_indices)
        .def_readwrite("kernel", &GaussianProcess::kernel)
        .def_readwrite("hp_opt_meta", &GaussianProcess::hp_opt_meta);

    py::class_<Kernel, std::shared_ptr<Kernel>>(m, "Kernel");

    py::class_<Hyperparameters, std::shared_ptr<Hyperparameters>>(m, "Hyperparameters");

    py::class_<SEHyperparams, Hyperparameters, std::shared_ptr<SEHyperparams>>(m, "SEHyperparams")
        .def(py::init<double, double, double>(), py::arg("signal_stdv"), py::arg("lengthscale"), py::arg("noise_stdv"))
        .def_readwrite("signal_stdv", &SEHyperparams::signal_stdv)
        .def_readwrite("lengthscale", &SEHyperparams::lengthscale)
        .def_readwrite("noise_stdv", &SEHyperparams::noise_stdv);

    py::class_<MaternHyperparams, Hyperparameters, std::shared_ptr<MaternHyperparams>>(m, "MaternHyperparams")
        .def(py::init<double, double, double, MaternSmoothness>(), 
        py::arg("signal_stdv"), py::arg("lengthscale"), py::arg("noise_stdv"), py::arg("nu"))
        .def_readwrite("signal_stdv", &MaternHyperparams::signal_stdv)
        .def_readwrite("lengthscale", &MaternHyperparams::lengthscale)
        .def_readwrite("noise_stdv", &MaternHyperparams::noise_stdv)
        .def_readwrite("nu", &MaternHyperparams::nu);

    py::class_<MaternExtendedHyperparams, MaternHyperparams, std::shared_ptr<MaternExtendedHyperparams>>(m, "MaternExtendedHyperparams")
        .def(py::init<double, double, double, MaternSmoothness, std::vector<double>>(), 
        py::arg("signal_stdv"), py::arg("lengthscale"), py::arg("noise_stdv"), py::arg("nu"), py::arg("override_noise_stdv"))
        .def_readwrite("override_noise_stdv", &MaternExtendedHyperparams::override_noise_stdv);

    py::class_<Matern, Kernel, std::shared_ptr<Matern>>(m, "Matern")
        .def(py::init<std::shared_ptr<Hyperparameters>, std::shared_ptr<Hyperparameters>, std::shared_ptr<Hyperparameters>, Eigen::VectorXi>(), py::arg("hyperparameters"), py::arg("lower_bounds"), py::arg("upper_bounds"), py::arg("conditional_indices"))
        .def("get_hyperparameters", &Matern::GetHyperparameters)
        .def("get_hp_lower_bounds", &Matern::GetHpLowerBounds)
        .def("get_hp_upper_bounds", &Matern::GetHpUpperBounds)
        .def("set_hyperparameters", &Matern::SetHyperparameters)
        .def("set_hp_lower_bounds", &Matern::SetHpLowerBounds)
        .def("set_hp_upper_bounds", &Matern::SetHpUpperBounds)
        .def("get_conditional_indices", &Matern::GetConditionalIndices);

    py::class_<SquaredExponential, Kernel, std::shared_ptr<SquaredExponential>>(m, "SquaredExponential")
        .def(py::init<std::shared_ptr<Hyperparameters>, std::shared_ptr<Hyperparameters>, std::shared_ptr<Hyperparameters>, Eigen::VectorXi>(), py::arg("hyperparameters"), py::arg("lower_bounds"), py::arg("upper_bounds"), py::arg("conditional_indices"))
        .def("get_hyperparameters", &SquaredExponential::GetHyperparameters)
        .def("get_hp_lower_bounds", &SquaredExponential::GetHpLowerBounds)
        .def("get_hp_upper_bounds", &SquaredExponential::GetHpUpperBounds)
        .def("set_hyperparameters", &SquaredExponential::SetHyperparameters)
        .def("set_hp_lower_bounds", &SquaredExponential::SetHpLowerBounds)
        .def("set_hp_upper_bounds", &SquaredExponential::SetHpUpperBounds)
        .def("get_conditional_indices", &SquaredExponential::GetConditionalIndices)
        .def("set_conditional_indices", &SquaredExponential::SetConditionalIndices);

    py::class_<MaternExtended, Matern, std::shared_ptr<MaternExtended>>(m, "MaternExtended")
        .def(py::init<std::shared_ptr<Hyperparameters>, std::shared_ptr<Hyperparameters>, std::shared_ptr<Hyperparameters>, Eigen::VectorXi, Eigen::VectorXi>(),
            py::arg("hyperparameters"),
            py::arg("lower_bounds"),
            py::arg("upper_bounds"),
            py::arg("conditional_indices"),
            py::arg("override_conditional_indices"))    
        .def("get_hyperparameters", &MaternExtended::GetHyperparameters)
        .def("get_hp_lower_bounds", &MaternExtended::GetHpLowerBounds)
        .def("get_hp_upper_bounds", &MaternExtended::GetHpUpperBounds)
        .def("set_hyperparameters", &MaternExtended::SetHyperparameters)
        .def("set_hp_lower_bounds", &MaternExtended::SetHpLowerBounds)
        .def("set_hp_upper_bounds", &MaternExtended::SetHpUpperBounds)
        .def("get_conditional_indices", &MaternExtended::GetConditionalIndices)
        .def("set_conditional_indices", &MaternExtended::SetConditionalIndices);

    py::enum_<MaternSmoothness>(m, "MaternSmoothness")
        .value("NU_0_5", MaternSmoothness::NU_0_5)
        .value("NU_1_5", MaternSmoothness::NU_1_5)
        .value("NU_2_5", MaternSmoothness::NU_2_5)
        .export_values();
}
