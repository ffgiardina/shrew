#include "shrew/gaussian_process/gaussian_process.hpp"

using namespace shrew::random_variable;
using namespace shrew::random_vector;

namespace gaussian_process {
    double GaussianProcess::LogMarginalLikelihood() {
        int m = y.size();
        auto K = kernel->OptimizationKernelFunc(kernel->GetOptimizationParams(), x(conditional_indices));
        Eigen::LLT<Eigen::MatrixXd> llt(K);
        Eigen::MatrixXd L = llt.matrixL();
        Eigen::VectorXd alpha = L.transpose().fullPivHouseholderQr().solve(L.fullPivHouseholderQr().solve(y));
        double log_trace = 0;
        for (int i = 0; i < m; ++i) {
            log_trace += log(L(i, i));
        }

        return -0.5 * y.transpose()*alpha - log_trace - m/2.0 * log(2 * M_PI);
    };

    double GaussianProcess::LogMarginalLikelihood(const std::vector<double> &params, Eigen::VectorXd &x_, Eigen::VectorXd &y_, kernel::Kernel *kernel_) {
        int m = y_.size();
        auto K = kernel_->OptimizationKernelFunc(params, x_);
        Eigen::LLT<Eigen::MatrixXd> llt(K);
        Eigen::MatrixXd L = llt.matrixL();
        
        Eigen::VectorXd alpha = L.transpose().fullPivHouseholderQr().solve(L.fullPivHouseholderQr().solve(y_));
        double log_trace = 0;
        for (int i = 0; i < m; ++i) {
            log_trace += log(L(i, i));
        }

        return -0.5 * y_.transpose()*alpha - log_trace - m/2.0 * log(2 * M_PI);
    };

    double GaussianProcess::LogMarginalLikelihood(const std::vector<double> &params, std::vector<double> &gradient, Eigen::VectorXd &x_, Eigen::VectorXd &y_, kernel::Kernel *kernel_) {
        int m = y_.size();
        auto K = kernel_->OptimizationKernelFunc(params, x_);
        Eigen::LLT<Eigen::MatrixXd> llt(K);
        Eigen::MatrixXd L = llt.matrixL();
        
        auto Linv = L.inverse();
        auto Kinv = Linv.transpose() * Linv;
        Eigen::VectorXd alpha = Kinv * y_;

        double log_trace = 0;
        for (int i = 0; i < m; ++i) {
            log_trace += log(L(i, i));
        }

        std::vector<Eigen::MatrixXd> dK = kernel_->OptimizationKernelDerivatives(params, x_);
        for (int i = 0; i < dK.size(); i++) {
            gradient[i] = 0.5 * ((alpha * alpha.transpose() - Kinv) * dK.at(i)).trace();
        }

        return -0.5 * y_.transpose()*alpha - log_trace - m/2.0 * log(2 * M_PI);
    };

    double GaussianProcess::lml_objective_func(const std::vector<double>& params, std::vector<double>& grad, void* data) {
        HpOptimizationMeta* hp_optim = reinterpret_cast<HpOptimizationMeta*>(data);

        double lml;
        if (!grad.empty()) {
            lml = LogMarginalLikelihood(params, grad, hp_optim->x_, hp_optim->y_, hp_optim->kernel_);
        }
        else {
            lml = LogMarginalLikelihood(params, hp_optim->x_, hp_optim->y_, hp_optim->kernel_);
        }
        
        hp_optim->running_max = lml > hp_optim->running_max ? lml : hp_optim->running_max;
        if (hp_optim->progress_output && hp_optim->iter % hp_optim->optim_progress_intervall == 0) {
            std::cout << "Iteration " << hp_optim->iter << ": " << hp_optim->running_max << std::endl;
            std::cout << "\tx: ";
            for (const auto& element : params) {
                std::cout << element << " ";
            }
            std::cout << std::endl;
        }
        hp_optim->iter += 1;

        return lml;
    }

    void GaussianProcess::OptimizeHyperparameters(bool progress_output) {
        hp_opt_meta.progress_output = progress_output;
        nlopt::opt opt(hp_opt_meta.algo, kernel->GetOptimizationParams().size());
        opt.set_lower_bounds(kernel->GetOptLowerBounds());
        opt.set_upper_bounds(kernel->GetOptUpperBounds());
        opt.set_ftol_rel(1e-6);
        opt.set_max_objective(lml_objective_func, &hp_opt_meta);
        kernel->ApplyParams(opt.optimize(kernel->GetOptimizationParams()));

        if (progress_output) {
            std::cout << "Optimization terminated @ ";
            std::cout << "iteration " << hp_opt_meta.iter << std::endl;
            std::cout << "Optimized Hyperparameters: ";
            for (const auto& element : kernel->GetOptimizationParams()) {
                std::cout << element << " ";
            }
            std::cout << "\nOptimized log maginal likelihood: " << LogMarginalLikelihood() << std::endl;
        }
    }

    std::tuple<std::vector<double>, std::vector<double>> GaussianProcess::GetPosteriorGP(){
        auto mu0 = Eigen::VectorXd::Zero(x.size());
        Eigen::MatrixXd K = kernel->KernelFunc(x);
        MultivariateNormal grv = MultivariateNormal(mu0, K);
        MultivariateNormal cgrv = getConditional(grv, conditional_indices, '=', y);

        int n = x.size() - y.size();
        std::vector<double> mean(n);
        std::vector<double> standard_deviation(n);
        for (int i = 0; i < n; ++i) {
            mean[i] = cgrv.mu(i);
            standard_deviation[i] = sqrt(cgrv.K(i, i));
        }
        return {mean, standard_deviation};
    }
}