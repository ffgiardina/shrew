#pragma once

#include <vector>
#include <Eigen/Dense>
#include <map>
#include "kernel.hpp"
#include "maternKernel.hpp"
#include "shrew/utils/cast_utils.hpp"

namespace gaussian_process {
    namespace kernel {

        struct MaternExtendedHyperparams : MaternHyperparams {
            std::vector<double> ext_data_noise_stdv;
        };

        struct MaternExtendedHyperparamsPartialDerivative : MaternHyperparamsPartialDerivative{
            std::vector<Eigen::MatrixXd> ext_data_noise_stdv;

            MaternExtendedHyperparamsPartialDerivative(int size, int ext_data_points): MaternHyperparamsPartialDerivative(size) {
                lengthscale = Eigen::MatrixXd::Zero(size, size);
                noise_stdv = Eigen::MatrixXd::Zero(size, size);
                signal_stdv = Eigen::MatrixXd::Zero(size, size);

                for (int i = 0; i < ext_data_points; ++i) {
                    ext_data_noise_stdv.push_back(Eigen::MatrixXd::Zero(size, size));
                }
            };
        };

        class MaternExtended : public Matern {
          private: 
            MaternExtendedHyperparams ext_hyperparameters;
            MaternExtendedHyperparams ext_hp_lower_bounds;
            MaternExtendedHyperparams ext_hp_upper_bounds;
            std::map<int, double> ext_data_conditional_indices_to_noise;

          public:
            virtual Eigen::MatrixXd KernelFunc(Eigen::VectorXd x) const override;
            virtual Eigen::MatrixXd DataKernelFunc(std::shared_ptr<Hyperparameters> ext_hyperparams, Eigen::VectorXd x) const override;
            virtual std::shared_ptr<HyperparametersPartialDerivative> DataKernelDerivatives(std::shared_ptr<Hyperparameters> ext_hyperparams, Eigen::VectorXd x) const override;
           
            std::map<int, double> GetExtConditionalIndices() const {return ext_data_conditional_indices_to_noise;};
            void SetExtConditionalIndices(std::map<int, double> cind) {ext_data_conditional_indices_to_noise = cind;};

            virtual std::shared_ptr<Hyperparameters> GetHyperparameters() const override {return std::make_shared<MaternExtendedHyperparams>(ext_hyperparameters);};
            virtual std::shared_ptr<Hyperparameters> GetHpLowerBounds() const override {return std::make_shared<MaternExtendedHyperparams>(ext_hp_lower_bounds);};
            virtual std::shared_ptr<Hyperparameters> GetHpUpperBounds() const override {return std::make_shared<MaternExtendedHyperparams>(ext_hp_upper_bounds);};
            virtual void SetHyperparameters(std::shared_ptr<Hyperparameters> hp) override {ext_hyperparameters = *shrew::utils::cast_pointer<MaternExtendedHyperparams>(hp);};
            virtual void SetHpLowerBounds(std::shared_ptr<Hyperparameters> hp_lb) override {ext_hp_lower_bounds = *shrew::utils::cast_pointer<MaternExtendedHyperparams>(hp_lb);};
            virtual void SetHpUpperBounds(std::shared_ptr<Hyperparameters> hp_ub) override {ext_hp_upper_bounds = *shrew::utils::cast_pointer<MaternExtendedHyperparams>(hp_ub);};

            MaternExtended(std::shared_ptr<Hyperparameters> hyperparameters_) : Matern(hyperparameters_) {
                ext_hyperparameters = *shrew::utils::cast_pointer<MaternExtendedHyperparams>(hyperparameters_, "shrew::kernel::MaternExtendedKernel(): hyperparameters_");
                ext_hp_lower_bounds = *shrew::utils::cast_pointer<MaternExtendedHyperparams>(hyperparameters_, "shrew::kernel::MaternExtendedKernel(): hyperparameters_");
                ext_hp_upper_bounds = *shrew::utils::cast_pointer<MaternExtendedHyperparams>(hyperparameters_, "shrew::kernel::MaternExtendedKernel(): hyperparameters_");
                ext_data_conditional_indices_to_noise = std::map<int, double> {};
            };
            MaternExtended(std::shared_ptr<Hyperparameters> hyperparameters_, std::shared_ptr<Hyperparameters> hp_lower_bounds_, 
                std::shared_ptr<Hyperparameters> hp_upper_bounds_, std::vector<int> conditional_indices_, std::vector<int> ext_data_conditional_indices_) : Matern(hyperparameters_, hp_lower_bounds_, hp_upper_bounds_, conditional_indices_) {
                ext_hyperparameters = *shrew::utils::cast_pointer<MaternExtendedHyperparams>(hyperparameters_, "shrew::kernel::MaternExtendedKernel(): hyperparameters_");
                ext_hp_lower_bounds = *shrew::utils::cast_pointer<MaternExtendedHyperparams>(hp_lower_bounds_, "shrew::kernel::MaternExtendedKernel(): hp_lower_bounds_");
                ext_hp_upper_bounds = *shrew::utils::cast_pointer<MaternExtendedHyperparams>(hp_upper_bounds_, "shrew::kernel::MaternExtendedKernel(): hp_upper_bounds");
                for (size_t i = 0; i < ext_hyperparameters.ext_data_noise_stdv.size(); ++i) {
                    ext_data_conditional_indices_to_noise[ext_data_conditional_indices_[i]] = ext_hyperparameters.ext_data_noise_stdv[i];
                }      
            };
        };
    }
}