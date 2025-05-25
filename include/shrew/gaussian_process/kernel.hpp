#pragma once

#include <Eigen/Dense>
#include <unordered_set>

namespace gaussian_process {
    namespace kernel {

        struct Hyperparameters {
            virtual ~Hyperparameters() = default;
        };
        
        class Kernel {
          public:
            virtual ~Kernel() = default;

            virtual Eigen::MatrixXd KernelFunc(Eigen::VectorXd x) const = 0;
            
            virtual std::unordered_set<int> GetConditionalIndices() const = 0;
            virtual void SetConditionalIndices(std::unordered_set<int>  cind) = 0;

            virtual std::shared_ptr<Hyperparameters> GetHyperparameters() const = 0;
            virtual std::shared_ptr<Hyperparameters> GetHpLowerBounds() const = 0;
            virtual std::shared_ptr<Hyperparameters> GetHpUpperBounds() const = 0;
            virtual void SetHyperparameters(std::shared_ptr<Hyperparameters> hp) = 0;
            virtual void SetHpLowerBounds(std::shared_ptr<Hyperparameters> hp_lb) = 0;
            virtual void SetHpUpperBounds(std::shared_ptr<Hyperparameters> hp_ub) = 0;

            virtual std::vector<double> GetOptimizationParams() const = 0;
            virtual std::vector<double> GetOptLowerBounds() const = 0;
            virtual std::vector<double> GetOptUpperBounds() const = 0;
            virtual void ApplyParams(const std::vector<double> &params) = 0;
            virtual Eigen::MatrixXd OptimizationKernelFunc(const std::vector<double> &opt_params_to_idx, const Eigen::VectorXd &x) const = 0;
            virtual std::vector<Eigen::MatrixXd> OptimizationKernelDerivatives(const std::vector<double> &opt_params_to_idx, const Eigen::VectorXd &x) const = 0;
        };
    }
}