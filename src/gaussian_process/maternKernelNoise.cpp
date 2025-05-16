#include "shrew/gaussian_process/maternKernelNoise.hpp"

namespace gaussian_process {
    namespace kernel {
        double MaternNoise::MaternEval(MaternSmoothness nu, double r, double l) {
            switch (nu) {
                case MaternSmoothness::NU_0_5: return exp(-r / l);
                case MaternSmoothness::NU_1_5: return (1.0 + sqrt(3) * r / l) * exp(- sqrt(3) * r / l);
                case MaternSmoothness::NU_2_5: return (1.0 + sqrt(5) * r / l + 5.0 * pow(r, 2.0) / (3.0 * pow(l, 2.0) )) * exp(- sqrt(5) * r / l);
                default: throw std::invalid_argument("Smoothness parameter value not supported");
            }
        }

        double MaternNoise::DerivativeMaternEval(MaternSmoothness nu, double r, double l) {
            switch (nu) {
                case MaternSmoothness::NU_0_5: return r / pow(l, 2.0) * exp(-r/l);
                case MaternSmoothness::NU_1_5: return (3 * exp(-(sqrt(3) * r)/l) * pow(r, 2.0))/ pow(l, 3.0);
                case MaternSmoothness::NU_2_5: return ( 5.0 / 3.0 * pow(r, 2.0) / pow(l, 3.0) + sqrt(5) * 5.0 / 3.0 * pow(r, 3.0) / pow(l, 4.0) ) * exp(- sqrt(5) * r / l);
                default: throw std::invalid_argument("Unknown smoothness parameter");
            }
        }

        Eigen::MatrixXd MaternNoise::KernelFunc(Eigen::VectorXd x) const {    
            Eigen::MatrixXd K(x.size(), x.size());
            for (int i = 0; i < x.size(); i++){
                for (int j = 0; j <= i; j++){
                    K(i, j) = pow(hyperparameters[0], 2) * MaternEval(nu, fabs(x(i) - x(j)), hyperparameters[1]);

                    if (i == j && conditional_idx_to_predictive_noise.contains(i)) 
                        K(i, j) += pow(conditional_idx_to_predictive_noise.at(i), 2);
                    else if (i == j && conditional_indices.contains(i)) 
                        K(i, j) += pow(hyperparameters[2], 2);
                    else if(i != j)
                        K(j, i) = K(i, j);
                }
            }
            return K;
        }

        Eigen::MatrixXd MaternNoise::DataKernelFunc(std::vector<double> hyperparams_, Eigen::VectorXd x) const {
            
            Eigen::MatrixXd K(x.size(), x.size());
            for (int i = 0; i < x.size(); i++){
                for (int j = 0; j <= i; j++){
                    K(i, j) = pow(hyperparams_[0], 2) * MaternEval(nu, fabs(x(i) - x(j)), hyperparams_[1]);
                    if (i == j && conditional_idx_to_predictive_noise.contains(i)) 
                        K(i, j) += pow(conditional_idx_to_predictive_noise.at(i), 2);
                    else if (i == j) 
                        K(i, j) += pow(hyperparams_[2], 2);
                    else 
                        K(j, i) = K(i, j);
                }
            }

            return K;
        }

        std::vector<Eigen::MatrixXd> MaternNoise::DataKernelDerivatives(std::vector<double> hyperparams_, Eigen::VectorXd x) const {
            std::vector<Eigen::MatrixXd> dK(hyperparams_.size());
            for (int i = 0; i < hyperparams_.size(); ++i) {
                dK[i] = Eigen::MatrixXd::Zero(x.size(), x.size());
            }
            for (int k = 0; k < hyperparams_.size(); k++) {
                for (int i = 0; i < x.size(); i++){
                    for (int j = 0; j <= i; j++){
                        if (k == 0) { 
                            dK[k](i, j) = 2 * hyperparams_[0] * MaternEval(nu, fabs(x(i) - x(j)), hyperparams_[1]);
                            if (i != j) 
                                dK[k](j, i) = dK[k](i, j);
                        } 
                        else if (k == 1) { 
                            dK[k](i, j) = pow(hyperparams_[0], 2) * DerivativeMaternEval(nu, fabs(x(i) - x(j)), hyperparams_[1]);
                            if (i != j)
                                dK[k](j, i) = dK[k](i, j);
                        }
                        else if (k == 2) {
                            dK[k](i, j) = 0;
                            if (i == j)
                                dK[k](i, j) += 2 * hyperparams_[2];
                            else
                                dK[k](j, i) = 0;
                        }
                        else if (k >= 3) {
                            dK[k](i, j) = 0;
                            if (i == j && conditional_idx_to_predictive_noise.contains(i)) {
                                dK[k](i, j) += 2 * conditional_idx_to_predictive_noise.at(i);
                            }
                            else{
                                dK[k](j, i) = 0;
                            }
                        }
                    }
                }
            }

            return dK;
        }
    }
}