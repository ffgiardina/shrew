#include "shrew/gaussian_process/squared_exponential_kernel.hpp"
#include "shrew/utils/cast_utils.hpp"

#include <iostream>

namespace gaussian_process {
    namespace kernel {
        std::vector<double> SquaredExponential::GetOptimizationParams() const {
            return {hyperparameters.signal_stdv, hyperparameters.lengthscale, hyperparameters.noise_stdv};
        }
        std::vector<double> SquaredExponential::GetOptLowerBounds() const {
            return {hp_lower_bounds.signal_stdv, hp_lower_bounds.lengthscale, hp_lower_bounds.noise_stdv};
        }
        std::vector<double> SquaredExponential::GetOptUpperBounds() const {
            return {hp_upper_bounds.signal_stdv, hp_upper_bounds.lengthscale, hp_upper_bounds.noise_stdv};
        }
        void SquaredExponential::ApplyParams(const std::vector<double> &params) {
            if (params.size() != 3) {
                throw std::invalid_argument("SquaredExponential::ApplyParams(): parameter vector must have size 3");
            }
            hyperparameters.signal_stdv = params[0];
            hyperparameters.lengthscale = params[1];
            hyperparameters.noise_stdv = params[2];
        }

        Eigen::MatrixXd SquaredExponential::KernelFunc(Eigen::VectorXd x) const {    
            Eigen::MatrixXd K(x.size(), x.size());
            for (int i = 0; i < x.size(); i++){
                for (int j = 0; j <= i; j++){
                    K(i, j) = pow(hyperparameters.signal_stdv, 2) * exp(-1.0 / (2.0 * pow(hyperparameters.lengthscale, 2.0)) * pow(x(i) - x(j), 2.0));

                    if (i == j && conditional_indices.contains(i))
                        K(i, j) += pow(hyperparameters.noise_stdv, 2);
                    else if(i != j)
                        K(j, i) = K(i, j);
                }
            }
            return K;
        }

        Eigen::MatrixXd SquaredExponential::OptimizationKernelFunc(const std::vector<double> &hyperparams, const Eigen::VectorXd &x) const {
            Eigen::MatrixXd K(x.size(), x.size());

            for (int i = 0; i < x.size(); i++){
                for (int j = 0; j <= i; j++){
                    K(i, j) = pow(hyperparams[opt_params_to_idx.at("signal_stdv")], 2) * exp(-1.0 / (2.0 * pow(hyperparams[opt_params_to_idx.at("lengthscale")], 2.0)) * pow(x(i) - x(j), 2.0));
                    if (i == j)
                        K(i, j) += pow(hyperparams[opt_params_to_idx.at("noise_stdv")], 2);
                    else
                        K(j, i) = K(i, j);
                }
            }

            return K;
        }

        std::vector<Eigen::MatrixXd> SquaredExponential::OptimizationKernelDerivatives(const std::vector<double> &hyperparams, const Eigen::VectorXd &x) const {
            std::vector<Eigen::MatrixXd> dK(hyperparams.size());
            for (int i = 0; i < hyperparams.size(); ++i) {
                dK[i] = Eigen::MatrixXd::Zero(x.size(), x.size());
            }

            for (int i = 0; i < x.size(); i++){
                for (int j = 0; j <= i; j++){
                        dK[opt_params_to_idx.at("signal_stdv")](i, j) = 2 * hyperparams[opt_params_to_idx.at("signal_stdv")] * exp(-1.0 / (2.0 * pow(hyperparams[opt_params_to_idx.at("lengthscale")], 2.0)) * pow(x(i) - x(j), 2.0));
                        if (i != j)
                            dK[opt_params_to_idx.at("signal_stdv")](j, i) = dK[opt_params_to_idx.at("signal_stdv")](i, j);

                        dK[opt_params_to_idx.at("lengthscale")](i, j) = 1.0 / pow(hyperparams[opt_params_to_idx.at("lengthscale")], 3.0) * pow(x(i) - x(j), 2.0) * pow(hyperparams[opt_params_to_idx.at("signal_stdv")], 2) * exp(-1.0 / (2.0 * pow(hyperparams[opt_params_to_idx.at("lengthscale")], 2.0)) * pow(x(i) - x(j), 2.0));
                        if (i != j)
                            dK[opt_params_to_idx.at("lengthscale")](j, i) = dK[opt_params_to_idx.at("lengthscale")](i, j);

                        dK[opt_params_to_idx.at("noise_stdv")](i, j) = 0;
                        if (i == j)
                            dK[opt_params_to_idx.at("noise_stdv")](i, j) += 2 * hyperparams[opt_params_to_idx.at("noise_stdv")];
                        else
                            dK[opt_params_to_idx.at("noise_stdv")](j, i) = 0;
                    }
                }
            
            return dK;
        }
    }
}