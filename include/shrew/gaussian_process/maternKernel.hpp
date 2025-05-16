#include <vector>
#include <Eigen/Dense>
#include <unordered_set>
#include "kernel.hpp"

namespace gaussian_process {
    namespace kernel {
        enum MaternSmoothness {
            NU_0_5,
            NU_1_5,
            NU_2_5
        };

        class Matern : public Kernel {
          private: 
            std::vector<double> hyperparameters; // {sigma_f, l, sigma_n}
            std::vector<double> hp_lower_bounds;
            std::vector<double> hp_upper_bounds;
            std::unordered_set<int> conditional_indices;
            static double MaternEval(MaternSmoothness nu, double r, double l);
            static double DerivativeMaternEval(MaternSmoothness nu, double r, double l);

          public:
            virtual Eigen::MatrixXd KernelFunc(Eigen::VectorXd x) const override;
            virtual Eigen::MatrixXd DataKernelFunc(std::vector<double> hyperparams, Eigen::VectorXd x) const override;
            virtual std::vector<Eigen::MatrixXd> DataKernelDerivatives(std::vector<double> hyperparams, Eigen::VectorXd x) const override;
          
            virtual std::unordered_set<int> GetConditionalIndices() const override {return conditional_indices;};
            virtual std::vector<double> GetHyperparameters() const override {return hyperparameters;};
            virtual std::vector<double> GetHpLowerBounds() const override {return hp_lower_bounds;};
            virtual std::vector<double> GetHpUpperBounds() const override {return hp_upper_bounds;};
            virtual void SetConditionalIndices(std::unordered_set<int> cind) override {conditional_indices = cind;};
            virtual void SetHyperparameters(std::vector<double> hp) override {hyperparameters = hp;};
            virtual void SetHpLowerBounds(std::vector<double> hp_lb) override {hp_lower_bounds = hp_lb;};
            virtual void SetHpUpperBounds(std::vector<double> hp_ub) override {hp_upper_bounds = hp_ub;};

            MaternSmoothness nu;

            Matern(std::vector<double> hyperparameters_, MaternSmoothness nu_) {
                hyperparameters = hyperparameters_;
                hp_lower_bounds = hyperparameters_;
                hp_upper_bounds = hyperparameters_;
                conditional_indices = std::unordered_set<int> {};
                nu = nu_;
            };
            Matern(std::vector<double> hyperparameters_, std::vector<double> hp_lower_bounds_, 
            std::vector<double> hp_upper_bounds_, std::vector<int> conditional_indices_, MaternSmoothness nu_) {
                hyperparameters = hyperparameters_;
                hp_lower_bounds = hp_lower_bounds_;
                hp_upper_bounds = hp_upper_bounds_;  
                conditional_indices.insert(conditional_indices_.begin(), conditional_indices_.end());      
                nu = nu_;    
                };
        };
    }
}