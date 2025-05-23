#pragma once

#include <vector>
#include <Eigen/Dense>
#include <unordered_set>
#include "kernel.hpp"
#include "shrew/utils/cast_utils.hpp"

namespace gaussian_process {
    namespace kernel {
        enum MaternSmoothness {
            NU_0_5,
            NU_1_5,
            NU_2_5
        };

        struct MaternHyperparams : Hyperparameters {
            double lengthscale;
            double noise_stdv;
            double signal_stdv;
            MaternSmoothness nu;
        };

        struct MaternHyperparamsPartialDerivative : HyperparametersPartialDerivative{
            Eigen::MatrixXd lengthscale;
            Eigen::MatrixXd noise_stdv;
            Eigen::MatrixXd signal_stdv;

            MaternHyperparamsPartialDerivative(int size) {
                lengthscale = Eigen::MatrixXd::Zero(size, size);
                noise_stdv = Eigen::MatrixXd::Zero(size, size);
                signal_stdv = Eigen::MatrixXd::Zero(size, size);
            };
        };

        class Matern : public Kernel {
          protected: 
            MaternHyperparams hyperparameters;
            MaternHyperparams hp_lower_bounds;
            MaternHyperparams hp_upper_bounds;

            std::unordered_set<int> conditional_indices;
            double MaternEval(double r, double l) const;
            double DerivativeMaternEval(double r, double l) const;

          public:
            virtual Eigen::MatrixXd KernelFunc(Eigen::VectorXd x) const override;
            virtual Eigen::MatrixXd DataKernelFunc(std::shared_ptr<Hyperparameters> hyperparams, Eigen::VectorXd x) const override;
            virtual std::shared_ptr<HyperparametersPartialDerivative> DataKernelDerivatives(std::shared_ptr<Hyperparameters> hyperparams, Eigen::VectorXd x) const override;
          
            virtual std::unordered_set<int> GetConditionalIndices() const override {return conditional_indices;};
            virtual std::shared_ptr<Hyperparameters> GetHyperparameters() const override {return std::make_shared<MaternHyperparams>(hyperparameters);};
            virtual std::shared_ptr<Hyperparameters> GetHpLowerBounds() const override {return std::make_shared<MaternHyperparams>(hp_lower_bounds);};
            virtual std::shared_ptr<Hyperparameters> GetHpUpperBounds() const override {return std::make_shared<MaternHyperparams>(hp_upper_bounds);};
            virtual void SetConditionalIndices(std::unordered_set<int> cind) override {conditional_indices = cind;};
            virtual void SetHyperparameters(std::shared_ptr<Hyperparameters> hp) override {hyperparameters = *shrew::utils::cast_pointer<MaternHyperparams>(hp);;};
            virtual void SetHpLowerBounds(std::shared_ptr<Hyperparameters> hp_lb) override {hp_lower_bounds = *shrew::utils::cast_pointer<MaternHyperparams>(hp_lb);;};
            virtual void SetHpUpperBounds(std::shared_ptr<Hyperparameters> hp_ub) override {hp_upper_bounds = *shrew::utils::cast_pointer<MaternHyperparams>(hp_ub);;};

            Matern(std::shared_ptr<Hyperparameters> hyperparameters_) {
                hyperparameters = *shrew::utils::cast_pointer<MaternHyperparams>(hyperparameters_, "shrew::kernel::MaternKernel(): hyperparameters_");
                hp_lower_bounds = *shrew::utils::cast_pointer<MaternHyperparams>(hyperparameters_, "shrew::kernel::MaternKernel(): hyperparameters_");
                hp_upper_bounds = *shrew::utils::cast_pointer<MaternHyperparams>(hyperparameters_, "shrew::kernel::MaternKernel(): hyperparameters_");
                conditional_indices = std::unordered_set<int> {};
            };
            Matern(std::shared_ptr<Hyperparameters> hyperparameters_, std::shared_ptr<Hyperparameters> hp_lower_bounds_, 
                std::shared_ptr<Hyperparameters> hp_upper_bounds_, std::vector<int> conditional_indices_) {
                hyperparameters = *shrew::utils::cast_pointer<MaternHyperparams>(hyperparameters_, "shrew::kernel::MaternKernel(): hyperparameters_");
                hp_lower_bounds = *shrew::utils::cast_pointer<MaternHyperparams>(hp_lower_bounds_, "shrew::kernel::MaternKernel(): hp_lower_bounds_");
                hp_upper_bounds = *shrew::utils::cast_pointer<MaternHyperparams>(hp_upper_bounds_, "shrew::kernel::MaternKernel(): hp_upper_bounds");
                conditional_indices.insert(conditional_indices_.begin(), conditional_indices_.end());      
                };
        };
    }
}