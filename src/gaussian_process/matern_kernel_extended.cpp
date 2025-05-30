#include "shrew/gaussian_process/matern_kernel_extended.hpp"

#include <iostream>

namespace gaussian_process {
    namespace kernel {

        std::vector<double> GetParamsVector(const MaternExtendedHyperparams &hp) {
            std::vector<double> params;
            params.push_back(hp.signal_stdv);
            params.push_back(hp.lengthscale);
            params.push_back(hp.noise_stdv);
            params.insert(params.end(), hp.override_noise_stdv.begin(), hp.override_noise_stdv.end());
            return params;
        }

        std::vector<double> MaternExtended::GetOptimizationParams() const {
            return GetParamsVector(ext_hyperparameters);
        }
        std::vector<double> MaternExtended::GetOptLowerBounds() const {
            return GetParamsVector(ext_hp_lower_bounds);
        }
        std::vector<double> MaternExtended::GetOptUpperBounds() const {
            return GetParamsVector(ext_hp_upper_bounds);
        }

        void MaternExtended::ApplyParams(const std::vector<double> &params) {
            if (params.size() != 3 + override_conditional_index_map.size()) {
                throw std::invalid_argument("MaternExtended::ApplyParams(): parameter vector must have size 3 + number of override noise data points");
            }
            ext_hyperparameters.signal_stdv = params.at(0);
            ext_hyperparameters.lengthscale = params.at(1);
            ext_hyperparameters.noise_stdv = params.at(2);
            ext_hyperparameters.override_noise_stdv.clear();
            for (size_t i = 3; i < params.size(); ++i) {
                ext_hyperparameters.override_noise_stdv.push_back(params.at(i));
            }
        }

        Eigen::MatrixXd MaternExtended::KernelFunc(Eigen::VectorXd x) const {    
            Eigen::MatrixXd K(x.size(), x.size());
            for (int i = 0; i < x.size(); i++){
                for (int j = 0; j <= i; j++){
                    K(i, j) = pow(ext_hyperparameters.signal_stdv, 2) * MaternEval(fabs(x(i) - x(j)), ext_hyperparameters.lengthscale);

                    if (i == j) {
                        if (override_joint_index_map.contains(i))
                            K(i, j) += pow(ext_hyperparameters.override_noise_stdv.at(override_joint_index_map.at(i)), 2);
                        else if (conditional_indices.contains(i))
                            K(i, j) += pow(ext_hyperparameters.noise_stdv, 2);
                    }
                    else
                        K(j, i) = K(i, j);
                }
            }
            return K;
        }

        Eigen::MatrixXd MaternExtended::OptimizationKernelFunc(const std::vector<double> &hyperparams, const Eigen::VectorXd &x) const {
            Eigen::MatrixXd K(x.size(), x.size());
            for (int i = 0; i < x.size(); i++){
                for (int j = 0; j <= i; j++){
                    K(i, j) = pow(hyperparams.at(opt_params_to_idx.at("signal_stdv")), 2) * MaternEval(fabs(x(i) - x(j)), hyperparams.at(opt_params_to_idx.at("lengthscale")));
                    
                    if (i == j) {
                        if (override_conditional_index_map.contains(i)) 
                            K(i, j) += pow(hyperparams.at(opt_params_to_idx.at("override_noise_stdv") + override_conditional_index_map.at(i)), 2);
                        else
                            K(i, j) += pow(hyperparams.at(opt_params_to_idx.at("noise_stdv")), 2);
                    }
                    else
                        K(j, i) = K(i, j);
                }
            }

            return K;
        }

        std::vector<Eigen::MatrixXd> MaternExtended::OptimizationKernelDerivatives(const std::vector<double> &hyperparams_, const Eigen::VectorXd &x) const {
            std::vector<Eigen::MatrixXd> dK(hyperparams_.size());
            for (int i = 0; i < hyperparams_.size(); ++i) {
                dK[i] = Eigen::MatrixXd::Zero(x.size(), x.size());
            }

            for (int i = 0; i < x.size(); i++){
                for (int j = 0; j <= i; j++){
                    dK[opt_params_to_idx.at("signal_stdv")](i, j) = 2 * hyperparams_.at(opt_params_to_idx.at("signal_stdv")) * MaternEval(fabs(x(i) - x(j)), hyperparams_.at(opt_params_to_idx.at("lengthscale")));
                    dK[opt_params_to_idx.at("lengthscale")](i, j) = pow(hyperparams_.at(opt_params_to_idx.at("signal_stdv")), 2) * DerivativeMaternEval(fabs(x(i) - x(j)), hyperparams_.at(opt_params_to_idx.at("lengthscale")));

                    if (i != j) {
                        dK[opt_params_to_idx.at("signal_stdv")](j, i) = dK.at(opt_params_to_idx.at("signal_stdv"))(i, j);
                        dK[opt_params_to_idx.at("lengthscale")](j, i) = dK.at(opt_params_to_idx.at("lengthscale"))(i, j);
                    }
                    else {
                        if (override_conditional_index_map.contains(i)) {
                            int k = opt_params_to_idx.at("override_noise_stdv") + override_conditional_index_map.at(i);
                            dK[k](i, j) += 2 * hyperparams_.at(k);
                        } else {
                            dK[opt_params_to_idx.at("noise_stdv")](i, j) += 2 * hyperparams_.at(opt_params_to_idx.at("noise_stdv"));
                        }
                    }
                }
            }

            return dK;
        }
    }
}