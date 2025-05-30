#pragma once

#include <iostream>
#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include <nlopt.hpp>
#include "kernel.hpp"
#include <shrew/multivariate_normal.hpp>
#include <shrew/random_variable.hpp>
#include <shrew/normal_distribution.hpp>

using namespace shrew::random_variable;
using namespace shrew::random_vector;

namespace gaussian_process {

    struct HpOptimizationMeta {
        int iter = 0;
        double running_max = -std::numeric_limits<double>::infinity();
        int optim_progress_intervall = 100;
        bool progress_output = true;
        nlopt::algorithm algo = nlopt::LD_LBFGS;
        Eigen::VectorXd y_;
        Eigen::VectorXd x_;
        kernel::Kernel *kernel_;
    };

    class GaussianProcess {
          public:
            Eigen::VectorXd y;
            Eigen::VectorXd x;
            Eigen::VectorXi conditional_indices;
            kernel::Kernel *kernel; 
            HpOptimizationMeta hp_opt_meta;

            double LogMarginalLikelihood();
            static double LogMarginalLikelihood(const std::vector<double> &params, Eigen::VectorXd &x_, Eigen::VectorXd &y_, kernel::Kernel *kernel_);
            static double LogMarginalLikelihood(const std::vector<double> &params, std::vector<double>& gradient, Eigen::VectorXd &x_, Eigen::VectorXd &y_, kernel::Kernel *kernel_);
            
            void OptimizeHyperparameters(bool progress_output = true);

            std::tuple<std::vector<double>, std::vector<double>> GetPosteriorGP();
            
            GaussianProcess(Eigen::VectorXd x_, Eigen::VectorXd y_,
              Eigen::VectorXi conditional_indices_, kernel::Kernel &kernel_) {
                x = x_;
                y = y_;
                conditional_indices = conditional_indices_;
                kernel = &kernel_;
                hp_opt_meta.x_ = x(conditional_indices);
                hp_opt_meta.y_ = y;
                hp_opt_meta.kernel_ = kernel;
            }

          private:    
            static double lml_objective_func(const std::vector<double>& params, std::vector<double>& grad, void* data);
    };
}