#include "shrew/gaussian_process/matern_kernel.hpp"

namespace gaussian_process {
    namespace kernel {
        std::vector<double> Matern::GetOptimizationParams() const {
            return {hyperparameters.signal_stdv, hyperparameters.lengthscale, hyperparameters.noise_stdv};
        }

        std::vector<double> Matern::GetOptLowerBounds() const {
            return {hp_lower_bounds.signal_stdv, hp_lower_bounds.lengthscale, hp_lower_bounds.noise_stdv};
        }

        std::vector<double> Matern::GetOptUpperBounds() const {
            return {hp_upper_bounds.signal_stdv, hp_upper_bounds.lengthscale, hp_upper_bounds.noise_stdv};
        }

        void Matern::ApplyParams(const std::vector<double> &params) {
            if (params.size() != 3) {
                throw std::invalid_argument("Matern::ApplyParams(): parameter vector must have size 3");
            }
            hyperparameters.signal_stdv = params[0];
            hyperparameters.lengthscale = params[1];
            hyperparameters.noise_stdv = params[2];
        }

        double Matern::MaternEval(double r, double l) const {
            switch (hyperparameters.nu) {
                case MaternSmoothness::NU_0_5: return exp(-r / l);
                case MaternSmoothness::NU_1_5: return (1.0 + sqrt(3) * r / l) * exp(- sqrt(3) * r / l);
                case MaternSmoothness::NU_2_5: return (1.0 + sqrt(5) * r / l + 5.0 * pow(r, 2.0) / (3.0 * pow(l, 2.0) )) * exp(- sqrt(5) * r / l);
                default: throw std::invalid_argument("Smoothness parameter value not supported");
            }
        }

        double Matern::DerivativeMaternEval(double r, double l) const {
            switch (hyperparameters.nu) {
                case MaternSmoothness::NU_0_5: return r / pow(l, 2.0) * exp(-r/l);
                case MaternSmoothness::NU_1_5: return (3 * exp(-(sqrt(3) * r)/l) * pow(r, 2.0))/ pow(l, 3.0);
                case MaternSmoothness::NU_2_5: return ( 5.0 / 3.0 * pow(r, 2.0) / pow(l, 3.0) + sqrt(5) * 5.0 / 3.0 * pow(r, 3.0) / pow(l, 4.0) ) * exp(- sqrt(5) * r / l);
                default: throw std::invalid_argument("Unknown smoothness parameter");
            }
        }

        Eigen::MatrixXd Matern::KernelFunc(Eigen::VectorXd x) const {    
            Eigen::MatrixXd K(x.size(), x.size());
            for (int i = 0; i < x.size(); i++){
                for (int j = 0; j <= i; j++){
                    K(i, j) = pow(hyperparameters.signal_stdv, 2) * MaternEval(fabs(x(i) - x(j)), hyperparameters.lengthscale);

                    if (i == j && conditional_indices.contains(i)) 
                        K(i, j) += pow(hyperparameters.noise_stdv, 2);
                    else if(i != j)
                        K(j, i) = K(i, j);
                }
            }
            return K;
        }

        Eigen::MatrixXd Matern::OptimizationKernelFunc(const std::vector<double> &hyperparams, const Eigen::VectorXd &x) const {
            Eigen::MatrixXd K(x.size(), x.size());
            for (int i = 0; i < x.size(); i++){
                for (int j = 0; j <= i; j++){
                    K(i, j) = pow(hyperparams[opt_params_to_idx.at("signal_stdv")], 2) * MaternEval(fabs(x(i) - x(j)), hyperparams[opt_params_to_idx.at("lengthscale")]);
                    if (i == j) 
                        K(i, j) += pow(hyperparams[opt_params_to_idx.at("noise_stdv")], 2);
                    else 
                        K(j, i) = K(i, j);
                }
            }

            return K;
        }

        std::vector<Eigen::MatrixXd> Matern::OptimizationKernelDerivatives(const std::vector<double> &hyperparams_, const Eigen::VectorXd &x) const {
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
                }
            }

            return dK;
        }
    }
}