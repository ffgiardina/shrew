#pragma once

#include <vector>
#include <Eigen/Dense>
#include <unordered_set>
#include <memory>
#include "kernel.hpp"
#include "shrew/utils/cast_utils.hpp"

namespace gaussian_process {
    namespace kernel {

        struct SEHyperparams : Hyperparameters {
          double lengthscale;
          double noise_stdv;
          double signal_stdv;

          SEHyperparams(double signal, double length, double noise)
          : lengthscale(length), noise_stdv(noise), signal_stdv(signal) {}
        };

        class SquaredExponential : public Kernel {
          private: 
            SEHyperparams hyperparameters;
            SEHyperparams hp_lower_bounds;
            SEHyperparams hp_upper_bounds;
            std::unordered_set<int> conditional_indices;

            const std::unordered_map<std::string, int> opt_params_to_idx = {
              {"signal_stdv", 0},
              {"lengthscale", 1},
              {"noise_stdv", 2}
            };
          
          public:
            virtual Eigen::MatrixXd KernelFunc(Eigen::VectorXd x) const override;
            virtual Eigen::MatrixXd OptimizationKernelFunc(const std::vector<double> &opt_params_to_idx, const Eigen::VectorXd &x) const override;
            virtual std::vector<Eigen::MatrixXd> OptimizationKernelDerivatives(const std::vector<double> &opt_params_to_idx, const Eigen::VectorXd &x) const override;

            virtual std::unordered_set<int> GetConditionalIndices() const override {return conditional_indices;};
            virtual void SetConditionalIndices(std::unordered_set<int> cind) override {conditional_indices = cind;};

            virtual std::shared_ptr<Hyperparameters> GetHyperparameters() const override {return std::make_shared<SEHyperparams>(hyperparameters);};
            virtual std::shared_ptr<Hyperparameters> GetHpLowerBounds() const override {return std::make_shared<SEHyperparams>(hp_lower_bounds);};
            virtual std::shared_ptr<Hyperparameters> GetHpUpperBounds() const override {return std::make_shared<SEHyperparams>(hp_upper_bounds);};
            virtual void SetHyperparameters(std::shared_ptr<Hyperparameters> hp) override {hyperparameters = *shrew::utils::cast_pointer<SEHyperparams>(hp);};
            virtual void SetHpLowerBounds(std::shared_ptr<Hyperparameters> hp_lb) override {hp_lower_bounds = *shrew::utils::cast_pointer<SEHyperparams>(hp_lb);};
            virtual void SetHpUpperBounds(std::shared_ptr<Hyperparameters> hp_ub) override {hp_upper_bounds = *shrew::utils::cast_pointer<SEHyperparams>(hp_ub);};

            virtual std::vector<double> GetOptimizationParams() const override;
            virtual std::vector<double> GetOptLowerBounds() const override;
            virtual std::vector<double> GetOptUpperBounds() const override;
            virtual void ApplyParams(const std::vector<double> &params) override;

            SquaredExponential(std::shared_ptr<Hyperparameters> hyperparameters_) : hyperparameters(*shrew::utils::cast_pointer<SEHyperparams>(hyperparameters_)),
              hp_lower_bounds(*shrew::utils::cast_pointer<SEHyperparams>(hyperparameters_)),
              hp_upper_bounds(*shrew::utils::cast_pointer<SEHyperparams>(hyperparameters_)) {
              conditional_indices = std::unordered_set<int> {};
            };

            SquaredExponential(std::shared_ptr<Hyperparameters> hyperparameters_, std::shared_ptr<Hyperparameters> hp_lower_bounds_, 
            std::shared_ptr<Hyperparameters> hp_upper_bounds_, Eigen::VectorXi conditional_indices_) : hyperparameters(*shrew::utils::cast_pointer<SEHyperparams>(hyperparameters_)),
            hp_lower_bounds(*shrew::utils::cast_pointer<SEHyperparams>(hp_lower_bounds_)),
            hp_upper_bounds(*shrew::utils::cast_pointer<SEHyperparams>(hp_upper_bounds_)) {
              conditional_indices = std::unordered_set<int>(conditional_indices_.begin(), conditional_indices_.end());
            };
        };
    }
}