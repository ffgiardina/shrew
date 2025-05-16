#pragma once

#include <vector>
#include <Eigen/Dense>
#include <unordered_set>

namespace gaussian_process {
    namespace kernel {

        class Kernel {
          public:
            virtual Eigen::MatrixXd KernelFunc(Eigen::VectorXd x) const = 0;
            virtual Eigen::MatrixXd DataKernelFunc(std::vector<double> hyperparams, Eigen::VectorXd x) const = 0;
            virtual std::vector<Eigen::MatrixXd> DataKernelDerivatives(std::vector<double> hyperparams, Eigen::VectorXd x) const = 0;
        
            virtual std::unordered_set<int> GetConditionalIndices() const = 0;
            virtual std::vector<double> GetHyperparameters() const = 0;
            virtual std::vector<double> GetHpLowerBounds() const = 0;
            virtual std::vector<double> GetHpUpperBounds() const = 0;
            virtual void SetConditionalIndices(std::unordered_set<int>  cind) = 0;
            virtual void SetHyperparameters(std::vector<double> hp) = 0;
            virtual void SetHpLowerBounds(std::vector<double> hp_lb) = 0;
            virtual void SetHpUpperBounds(std::vector<double> hp_ub) = 0;
        };
    }
}