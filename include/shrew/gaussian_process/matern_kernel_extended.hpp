#pragma once

#include <vector>
#include <Eigen/Dense>
#include <map>
#include <memory>
#include "kernel.hpp"
#include "matern_kernel.hpp"
#include "shrew/utils/cast_utils.hpp"

namespace gaussian_process {
    namespace kernel {

        struct MaternExtendedHyperparams : MaternHyperparams {
            std::vector<double> override_noise_stdv;

            MaternExtendedHyperparams(double signal, double length, double noise, MaternSmoothness nu_ = NU_1_5, std::vector<double> override_noise_stdv_ = {})
                : MaternHyperparams(signal, length, noise, nu_), override_noise_stdv(override_noise_stdv_) {}
        };

        // Extended Matern kernel that allows to override the noise hyperparameter of selected conditional indices with custom values.
        class MaternExtended : public Matern {
          private: 
            MaternExtendedHyperparams ext_hyperparameters;
            MaternExtendedHyperparams ext_hp_lower_bounds;
            MaternExtendedHyperparams ext_hp_upper_bounds;
            std::unordered_map<int, int> override_joint_index_map;
            std::unordered_map<int, int> override_conditional_index_map;

            std::unordered_map<std::string, int> opt_params_to_idx = {
                {"signal_stdv", 0},
                {"lengthscale", 1},
                {"noise_stdv", 2},
                {"override_noise_stdv", 3} // Extra data noise stdv starts at index 3
              };

          public:
            virtual Eigen::MatrixXd KernelFunc(Eigen::VectorXd x) const override;
            virtual Eigen::MatrixXd OptimizationKernelFunc(const std::vector<double> &opt_params_to_idx, const Eigen::VectorXd &x) const override;
            virtual std::vector<Eigen::MatrixXd> OptimizationKernelDerivatives(const std::vector<double> &opt_params_to_idx, const Eigen::VectorXd &x) const override;

            virtual std::shared_ptr<Hyperparameters> GetHyperparameters() const override {return std::make_shared<MaternExtendedHyperparams>(ext_hyperparameters);};
            virtual std::shared_ptr<Hyperparameters> GetHpLowerBounds() const override {return std::make_shared<MaternExtendedHyperparams>(ext_hp_lower_bounds);};
            virtual std::shared_ptr<Hyperparameters> GetHpUpperBounds() const override {return std::make_shared<MaternExtendedHyperparams>(ext_hp_upper_bounds);};
            virtual void SetHyperparameters(std::shared_ptr<Hyperparameters> hp) override {ext_hyperparameters = *shrew::utils::cast_pointer<MaternExtendedHyperparams>(hp);};
            virtual void SetHpLowerBounds(std::shared_ptr<Hyperparameters> hp_lb) override {ext_hp_lower_bounds = *shrew::utils::cast_pointer<MaternExtendedHyperparams>(hp_lb);};
            virtual void SetHpUpperBounds(std::shared_ptr<Hyperparameters> hp_ub) override {ext_hp_upper_bounds = *shrew::utils::cast_pointer<MaternExtendedHyperparams>(hp_ub);};

            virtual std::vector<double> GetOptimizationParams() const override;
            virtual std::vector<double> GetOptLowerBounds() const override;
            virtual std::vector<double> GetOptUpperBounds() const override;
            virtual void ApplyParams(const std::vector<double> &params) override;

            std::unordered_map<std::string, int> GetOptParamsToIdx() const {
                return opt_params_to_idx;
            }
            std::unordered_map<int, int> GetExtDataConditionalIndexMap() const {
                return override_conditional_index_map;
            }

            MaternExtended(std::shared_ptr<Hyperparameters> hyperparameters_) : Matern(hyperparameters_), 
            ext_hyperparameters(*shrew::utils::cast_pointer<MaternExtendedHyperparams>(hyperparameters_, "shrew::kernel::MaternExtendedKernel(): hyperparameters_")),
            ext_hp_lower_bounds(*shrew::utils::cast_pointer<MaternExtendedHyperparams>(hyperparameters_, "shrew::kernel::MaternExtendedKernel(): hyperparameters_")),
            ext_hp_upper_bounds(*shrew::utils::cast_pointer<MaternExtendedHyperparams>(hyperparameters_, "shrew::kernel::MaternExtendedKernel(): hyperparameters_")) {
                override_joint_index_map = std::unordered_map<int, int> {};
                override_conditional_index_map = std::unordered_map<int, int> {};
            };
            MaternExtended(std::shared_ptr<Hyperparameters> hyperparameters_, std::shared_ptr<Hyperparameters> hp_lower_bounds_, 
                std::shared_ptr<Hyperparameters> hp_upper_bounds_, std::vector<int> conditional_indices_, std::vector<int> override_conditional_indices_) : 
                Matern(hyperparameters_, hp_lower_bounds_, hp_upper_bounds_, conditional_indices_), 
                ext_hyperparameters(*shrew::utils::cast_pointer<MaternExtendedHyperparams>(hyperparameters_, "shrew::kernel::MaternExtendedKernel(): hyperparameters_")),
                ext_hp_lower_bounds(*shrew::utils::cast_pointer<MaternExtendedHyperparams>(hp_lower_bounds_, "shrew::kernel::MaternExtendedKernel(): hp_lower_bounds_")),
                ext_hp_upper_bounds(*shrew::utils::cast_pointer<MaternExtendedHyperparams>(hp_upper_bounds_, "shrew::kernel::MaternExtendedKernel(): hp_upper_bounds_")) {
                for (size_t i = 0; i < ext_hyperparameters.override_noise_stdv.size(); ++i) {
                    override_joint_index_map[conditional_indices_[override_conditional_indices_[i]]] = i;
                    override_conditional_index_map[override_conditional_indices_[i]] = i;
                }      
            };
        };
    }
}