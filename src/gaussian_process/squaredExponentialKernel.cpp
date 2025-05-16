#include "shrew/gaussian_process/squaredExponentialKernel.hpp"

namespace gaussian_process {
    namespace kernel {
        Eigen::MatrixXd SquaredExponential::KernelFunc(Eigen::VectorXd x) const {    
            Eigen::MatrixXd K(x.size(), x.size());
            for (int i = 0; i < x.size(); i++){
                for (int j = 0; j <= i; j++){
                    K(i, j) = pow(hyperparameters[0], 2) * exp(-1.0 / (2.0 * pow(hyperparameters[1], 2.0)) * pow(x(i) - x(j), 2.0));

                    if (i == j && conditional_indices.contains(i))
                        K(i, j) += pow(hyperparameters[2], 2);
                    else if(i != j)
                        K(j, i) = K(i, j);
                }
            }
            return K;
        }

        Eigen::MatrixXd SquaredExponential::DataKernelFunc(std::vector<double> hyperparams, Eigen::VectorXd x) const {
            Eigen::MatrixXd K(x.size(), x.size());
            for (int i = 0; i < x.size(); i++){
                for (int j = 0; j <= i; j++){
                    K(i, j) = pow(hyperparams[0], 2) * exp(-1.0 / (2.0 * pow(hyperparams[1], 2.0)) * pow(x(i) - x(j), 2.0));
                    if (i == j)
                        K(i, j) += pow(hyperparams[2], 2);
                    else
                        K(j, i) = K(i, j);
                }
            }

            return K;
        }

        std::vector<Eigen::MatrixXd> SquaredExponential::DataKernelDerivatives(std::vector<double> hyperparams, Eigen::VectorXd x) const {
            std::vector<Eigen::MatrixXd> dK(hyperparams.size());
            for (int i = 0; i < hyperparams.size(); ++i) {
                dK[i] = Eigen::MatrixXd::Zero(x.size(), x.size());
            }

            for (int k = 0; k < hyperparams.size(); k++) {
                for (int i = 0; i < x.size(); i++){
                    for (int j = 0; j <= i; j++){
                        if (k == 0) { 
                            dK[k](i, j) = 2 * hyperparams[0] * exp(-1.0 / (2.0 * pow(hyperparams[1], 2.0)) * pow(x(i) - x(j), 2.0));
                            if (i != j)
                                dK[k](j, i) = dK[k](i, j);
                        } 
                        else if (k == 1) { 
                            dK[k](i, j) = 1.0 / pow(hyperparams[1], 3.0) * pow(x(i) - x(j), 2.0) * pow(hyperparams[0], 2) * exp(-1.0 / (2.0 * pow(hyperparams[1], 2.0)) * pow(x(i) - x(j), 2.0));
                            if (i != j)
                                dK[k](j, i) = dK[k](i, j);
                        }
                        else if (k == 2) {
                            dK[k](i, j) = 0;
                            if (i == j)
                                dK[k](i, j) += 2 * hyperparams[2];
                            else
                                dK[k](j, i) = 0;
                        }
                    }
                }
            }

            return dK;
        }
    }
}