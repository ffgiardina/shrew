#pragma once

#include <Eigen/Dense>
#include <unordered_set>

namespace gaussian_process {
    namespace kernel {

        struct Hyperparameters {
            virtual ~Hyperparameters() = default;
        };

        struct HyperparametersPartialDerivative {
            virtual ~HyperparametersPartialDerivative() = default;
        };
        
        class Kernel {
          public:
            virtual ~Kernel() = default;

            virtual Eigen::MatrixXd KernelFunc(Eigen::VectorXd x) const = 0;
            virtual Eigen::MatrixXd DataKernelFunc(std::shared_ptr<Hyperparameters> hyperparams, Eigen::VectorXd x) const = 0;
            virtual std::shared_ptr<HyperparametersPartialDerivative> DataKernelDerivatives(std::shared_ptr<Hyperparameters> hyperparams, Eigen::VectorXd x) const = 0;
        
            virtual std::unordered_set<int> GetConditionalIndices() const = 0;
            virtual std::shared_ptr<Hyperparameters> GetHyperparameters() const = 0;
            virtual std::shared_ptr<Hyperparameters> GetHpLowerBounds() const = 0;
            virtual std::shared_ptr<Hyperparameters> GetHpUpperBounds() const = 0;
            virtual void SetConditionalIndices(std::unordered_set<int>  cind) = 0;
            virtual void SetHyperparameters(std::shared_ptr<Hyperparameters> hp) = 0;
            virtual void SetHpLowerBounds(std::shared_ptr<Hyperparameters> hp_lb) = 0;
            virtual void SetHpUpperBounds(std::shared_ptr<Hyperparameters> hp_ub) = 0;
        };
    }
}