#include "shrew/gaussian_process/matern_kernel_extended.hpp"

namespace gaussian_process {
    namespace kernel {

        std::vector<double> GetParamsVector(const MaternExtendedHyperparams &hp) {
            std::vector<double> params;
            params.push_back(hp.signal_stdv);
            params.push_back(hp.lengthscale);
            params.push_back(hp.noise_stdv);
            params.insert(params.end(), hp.ext_data_noise_stdv.begin(), hp.ext_data_noise_stdv.end());
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
            if (params.size() != 3 + ext_data_conditional_index_map.size()) {
                throw std::invalid_argument("MaternExtended::ApplyParams(): parameter vector must have size 3 + number of external data points");
            }
            ext_hyperparameters.signal_stdv = params[0];
            ext_hyperparameters.lengthscale = params[1];
            ext_hyperparameters.noise_stdv = params[2];
            ext_hyperparameters.ext_data_noise_stdv.clear();
            for (size_t i = 3; i < params.size(); ++i) {
                ext_hyperparameters.ext_data_noise_stdv.push_back(params[i]);
            }
        }

        Eigen::MatrixXd MaternExtended::KernelFunc(Eigen::VectorXd x) const {    
            Eigen::MatrixXd K(x.size(), x.size());
            for (int i = 0; i < x.size(); i++){
                for (int j = 0; j <= i; j++){
                    K(i, j) = pow(hyperparameters.signal_stdv, 2) * MaternEval(fabs(x(i) - x(j)), hyperparameters.lengthscale);

                    if (i == j && ext_data_conditional_index_map.contains(i)) 
                        K(i, j) += pow(ext_hyperparameters.ext_data_noise_stdv[ext_data_conditional_index_map.at(i)], 2);
                    else if (i == j && conditional_indices.contains(i)) 
                        K(i, j) += pow(hyperparameters.noise_stdv, 2);
                    else if(i != j)
                        K(j, i) = K(i, j);
                }
            }
            return K;
        }

        Eigen::MatrixXd MaternExtended::OptimizationKernelFunc(const std::vector<double> &hyperparams, const Eigen::VectorXd &x) const {
            Eigen::MatrixXd K(x.size(), x.size());
            for (int i = 0; i < x.size(); i++){
                for (int j = 0; j <= i; j++){
                    K(i, j) = pow(hyperparams[opt_params_to_idx.at("signal_stdv")], 2) * MaternEval(fabs(x(i) - x(j)), hyperparams[opt_params_to_idx.at("lengthscale")]);
                    
                    if (i == j && ext_data_conditional_index_map.contains(i)) 
                        K(i, j) += pow(hyperparams[n_base_params + ext_data_conditional_index_map.at(i)], 2);
                    else if (i == j) 
                        K(i, j) += pow(hyperparams[opt_params_to_idx.at("noise_stdv")], 2);
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
                    dK[opt_params_to_idx.at("signal_stdv")](i, j) = 2 * hyperparams_[opt_params_to_idx.at("signal_stdv")] * MaternEval(fabs(x(i) - x(j)), hyperparams_[opt_params_to_idx.at("lengthscale")]);
                    if (i != j) 
                        dK[opt_params_to_idx.at("signal_stdv")](j, i) = dK[opt_params_to_idx.at("signal_stdv")](i, j);

                    dK[opt_params_to_idx.at("lengthscale")](i, j) = pow(hyperparams_[opt_params_to_idx.at("signal_stdv")], 2) * DerivativeMaternEval(fabs(x(i) - x(j)), hyperparams_[opt_params_to_idx.at("lengthscale")]);
                    if (i != j)
                        dK[opt_params_to_idx.at("lengthscale")](j, i) = dK[opt_params_to_idx.at("lengthscale")](i, j);
                    
                    dK[opt_params_to_idx.at("noise_stdv")](i, j) = 0;
                    if (i == j)
                        dK[opt_params_to_idx.at("noise_stdv")](i, j) += 2 * hyperparams_[opt_params_to_idx.at("noise_stdv")];
                    else
                        dK[opt_params_to_idx.at("noise_stdv")](j, i) = 0;

                    for (size_t k = opt_params_to_idx.at("ext_data_noise_stdv"); k < hyperparams_.size(); ++k) {
                        dK[k](i, j) = 0;
                        if (i == j && ext_data_conditional_index_map.contains(i) && ext_data_conditional_index_map.at(i) == k - n_base_params) {
                            dK[k](i, j) += 2 * hyperparams_[k];
                        }
                        else
                            dK[k](j, i) = 0;
                    }
                }
            }

            return dK;
        }
    }
}