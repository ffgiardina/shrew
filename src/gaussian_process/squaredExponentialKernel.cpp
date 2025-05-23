#include "shrew/gaussian_process/squaredExponentialKernel.hpp"
#include "shrew/utils/cast_utils.hpp"

namespace gaussian_process {
    namespace kernel {
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

        Eigen::MatrixXd SquaredExponential::DataKernelFunc(std::shared_ptr<Hyperparameters> hyperparams_, Eigen::VectorXd x) const {
            Eigen::MatrixXd K(x.size(), x.size());
            std::shared_ptr<SEHyperparams> hyperparams = shrew::utils::cast_pointer<SEHyperparams>(hyperparams_, "SquaredExponential::DataKernelFunc()" );

            for (int i = 0; i < x.size(); i++){
                for (int j = 0; j <= i; j++){
                    K(i, j) = pow(hyperparams->signal_stdv, 2) * exp(-1.0 / (2.0 * pow(hyperparams->lengthscale, 2.0)) * pow(x(i) - x(j), 2.0));
                    if (i == j)
                        K(i, j) += pow(hyperparams->noise_stdv, 2);
                    else
                        K(j, i) = K(i, j);
                }
            }

            return K;
        }

        std::shared_ptr<HyperparametersPartialDerivative> SquaredExponential::DataKernelDerivatives(std::shared_ptr<Hyperparameters> hyperparams_, Eigen::VectorXd x) const {
            SEHyperparamsPartialDerivative dK = SEHyperparamsPartialDerivative(x.size());
            std::shared_ptr<SEHyperparams> hyperparams = shrew::utils::cast_pointer<SEHyperparams>(hyperparams_, "SquaredExponential::DataKernelDerivatives()" );

            for (int i = 0; i < x.size(); i++){
                for (int j = 0; j <= i; j++){
                        dK.signal_stdv(i, j) = 2 * hyperparams->signal_stdv * exp(-1.0 / (2.0 * pow(hyperparams->lengthscale, 2.0)) * pow(x(i) - x(j), 2.0));
                        if (i != j)
                            dK.signal_stdv(j, i) = dK.signal_stdv(i, j);

                        dK.lengthscale(i, j) = 1.0 / pow(hyperparams->lengthscale, 3.0) * pow(x(i) - x(j), 2.0) * pow(hyperparams->signal_stdv, 2) * exp(-1.0 / (2.0 * pow(hyperparams->lengthscale, 2.0)) * pow(x(i) - x(j), 2.0));
                        if (i != j)
                            dK.lengthscale(j, i) = dK.lengthscale(i, j);

                        dK.noise_stdv(i, j) = 0;
                        if (i == j)
                            dK.noise_stdv(i, j) += 2 * hyperparams->noise_stdv;
                        else
                            dK.noise_stdv(j, i) = 0;
                    }
                }
            
            return std::make_shared<SEHyperparamsPartialDerivative>(dK);
        }
    }
}