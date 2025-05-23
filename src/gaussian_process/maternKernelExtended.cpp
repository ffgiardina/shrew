#include "shrew/gaussian_process/maternKernelExtended.hpp"

namespace gaussian_process {
    namespace kernel {
        Eigen::MatrixXd MaternExtended::KernelFunc(Eigen::VectorXd x) const {    
            Eigen::MatrixXd K(x.size(), x.size());
            for (int i = 0; i < x.size(); i++){
                for (int j = 0; j <= i; j++){
                    K(i, j) = pow(hyperparameters.signal_stdv, 2) * MaternEval(fabs(x(i) - x(j)), hyperparameters.lengthscale);

                    if (i == j && ext_data_conditional_indices_to_noise.contains(i)) 
                        K(i, j) += pow(ext_data_conditional_indices_to_noise.at(i), 2);
                    else if (i == j && conditional_indices.contains(i)) 
                        K(i, j) += pow(hyperparameters.noise_stdv, 2);
                    else if(i != j)
                        K(j, i) = K(i, j);
                }
            }
            return K;
        }

        Eigen::MatrixXd MaternExtended::DataKernelFunc(std::shared_ptr<Hyperparameters> hyperparams, Eigen::VectorXd x) const {
            std::shared_ptr<MaternHyperparams> hyperparams_ = shrew::utils::cast_pointer<MaternHyperparams>(hyperparams, "SquaredExponential::DataKernelFunc()" );
            Eigen::MatrixXd K(x.size(), x.size());
            for (int i = 0; i < x.size(); i++){
                for (int j = 0; j <= i; j++){
                    K(i, j) = pow(hyperparams_->signal_stdv, 2) * MaternEval(fabs(x(i) - x(j)), hyperparams_->lengthscale);
                    
                    if (i == j && ext_data_conditional_indices_to_noise.contains(i)) 
                        K(i, j) += pow(ext_data_conditional_indices_to_noise.at(i), 2);
                    else if (i == j) 
                        K(i, j) += pow(hyperparams_->noise_stdv, 2);
                    else 
                        K(j, i) = K(i, j);
                }
            }

            return K;
        }

        std::shared_ptr<HyperparametersPartialDerivative> MaternExtended::DataKernelDerivatives(std::shared_ptr<Hyperparameters> hyperparams, Eigen::VectorXd x) const {
            std::shared_ptr<MaternHyperparams> hyperparams_ = shrew::utils::cast_pointer<MaternHyperparams>(hyperparams, "Matern::DataKernelDerivatives()" );

            auto dK = MaternExtendedHyperparamsPartialDerivative(x.size(), ext_data_conditional_indices_to_noise.size());
            for (int i = 0; i < x.size(); i++){
                for (int j = 0; j <= i; j++){
                        dK.signal_stdv(i, j) = 2 * hyperparams_->signal_stdv * MaternEval(fabs(x(i) - x(j)), hyperparams_->lengthscale);
                        if (i != j) 
                            dK.signal_stdv(j, i) = dK.signal_stdv(i, j);

                        dK.lengthscale(i, j) = pow(hyperparams_->signal_stdv, 2) * DerivativeMaternEval(fabs(x(i) - x(j)), hyperparams_->lengthscale);
                        if (i != j)
                            dK.lengthscale(j, i) = dK.lengthscale(i, j);
                        
                        dK.noise_stdv(i, j) = 0;
                        if (i == j)
                            dK.noise_stdv(i, j) += 2 * hyperparams_->noise_stdv;
                        else
                            dK.noise_stdv(j, i) = 0;


                        int k = 0;
                        for (auto & ext_idx : ext_data_conditional_indices_to_noise) {
                            dK.ext_data_noise_stdv[k](i, j) = 0;
                            if (i == j && ext_idx.first == i) {
                                dK.ext_data_noise_stdv[k](i, j) += 2 * ext_idx.second;
                            }
                            else
                                dK.ext_data_noise_stdv[k](j, i) = 0;
                            k++;
                        }
                }
            }

            return std::make_shared<MaternHyperparamsPartialDerivative>(dK);
        }
    }
}