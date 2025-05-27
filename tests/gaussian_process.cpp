#include "shrew/gaussian_process/gaussian_process.hpp"
#include "shrew/gaussian_process/matern_kernel.hpp"
#include "shrew/gaussian_process/squared_exponential_kernel.hpp"
#include "shrew/gaussian_process/matern_kernel_extended.hpp"

#include <vector>
#include <utility>
#include <cmath>
#include <random>
#include <gtest/gtest.h>

using namespace gaussian_process;

class GaussianProcessTestFixture : public testing::Test
{
protected:
    static constexpr int num_data = 50;
    static constexpr int num_eval = 100;
    Eigen::VectorXd x_data = Eigen::VectorXd::LinSpaced(num_data, 0, 2 * M_PI);
    Eigen::VectorXd x_eval = Eigen::VectorXd::Zero(num_eval);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(num_data + num_eval);
    Eigen::VectorXd y = Eigen::VectorXd::Zero(num_data);
    std::vector<int> conditional_indices;
    const double range_x = 10.0;
    const double noise_std = 0.2;
    const double freq = 1.0;
    const std::vector<double> ext_data_noise_stdv = {0.1, 0.2, 0.3, 0.6};
    const std::vector<int> ext_data_conditional_indices = {5, 12, 15, 20};

    void SetUp() override {
        generateNoisySinusoid();
    }

    void generateNoisySinusoid() {
        std::default_random_engine generator(42);
        std::normal_distribution<double> distribution(0.0, noise_std);

        conditional_indices.resize(num_data);
        for (int i = 0; i < num_data; ++i) {
            conditional_indices[i] = i + num_eval;
        }

        for (int i = 0; i < num_data; ++i) {
            x_data[i] = i * range_x / num_data;
            y[i] = std::sin(x_data[i] * freq) + distribution(generator);
        }
        for (int i = 0; i < num_eval; ++i) {
            x_eval[i] = i * range_x / num_eval;
        }
        x << x_eval, x_data;
    }
};

TEST_F(GaussianProcessTestFixture, SuqaredExponentialKernelOptimizationTest) {
    auto hyperparams = std::make_shared<kernel::SEHyperparams>(kernel::SEHyperparams(0.5, 0.3, 0.5));
    auto hyperparams_lower = std::make_shared<kernel::SEHyperparams>(kernel::SEHyperparams(0.01, 0.01, 0.01));
    auto hyperparams_upper = std::make_shared<kernel::SEHyperparams>(kernel::SEHyperparams(1000.0, 1000.0, 1000.0));

    kernel::SquaredExponential kernel(hyperparams, hyperparams_lower, hyperparams_upper, conditional_indices);
    GaussianProcess gp(x, y, conditional_indices, kernel);

    auto init_log_marginal_likelihood = gp.LogMarginalLikelihood();
    EXPECT_NO_THROW(gp.OptimizeHyperparameters(false));
    EXPECT_NO_THROW(gp.LogMarginalLikelihood());
    EXPECT_NO_THROW(gp.GetPosteriorGP());
    EXPECT_GT(gp.LogMarginalLikelihood(), init_log_marginal_likelihood) << "Log marginal likelihood did not improve after optimization";
}

TEST_F(GaussianProcessTestFixture, MaternKernelOptimizationTest) {
    auto hyperparams = std::make_shared<kernel::MaternHyperparams>(kernel::MaternHyperparams(0.5, 0.3, 0.5, kernel::MaternSmoothness::NU_1_5));
    auto hyperparams_lower = std::make_shared<kernel::MaternHyperparams>(kernel::MaternHyperparams(0.01, 0.01, 0.01, kernel::MaternSmoothness::NU_1_5));
    auto hyperparams_upper = std::make_shared<kernel::MaternHyperparams>(kernel::MaternHyperparams(1000.0, 1000.0, 1000.0, kernel::MaternSmoothness::NU_1_5));

    gaussian_process::kernel::Matern kernel(hyperparams, hyperparams_lower, hyperparams_upper, conditional_indices);
    gaussian_process::GaussianProcess gp(x, y, conditional_indices, kernel);

    auto init_log_marginal_likelihood = gp.LogMarginalLikelihood();
    EXPECT_NO_THROW(gp.OptimizeHyperparameters(false));
    EXPECT_NO_THROW(gp.LogMarginalLikelihood());
    EXPECT_NO_THROW(gp.GetPosteriorGP());
    EXPECT_GT(gp.LogMarginalLikelihood(), init_log_marginal_likelihood) << "Log marginal likelihood did not improve after optimization";
}

TEST_F(GaussianProcessTestFixture, MaternKernelExtendedLogicTest) {
    const std::vector<double> ext_data_noise_stdv_lb = {0.01, 0.01, 0.01, 0.01, 0.01};
    const std::vector<double> ext_data_noise_stdv_ub = {1000.0, 1000.0, 1000.0, 1000.0, 1000.0};

    auto hyperparams = std::make_shared<kernel::MaternExtendedHyperparams>(kernel::MaternExtendedHyperparams(0.5, 0.3, 0.5, kernel::MaternSmoothness::NU_1_5, ext_data_noise_stdv));
    auto hyperparams_lower = std::make_shared<kernel::MaternExtendedHyperparams>(kernel::MaternExtendedHyperparams(0.01, 0.01, 0.01, kernel::MaternSmoothness::NU_1_5, ext_data_noise_stdv_lb));
    auto hyperparams_upper = std::make_shared<kernel::MaternExtendedHyperparams>(kernel::MaternExtendedHyperparams(1000.0, 1000.0, 1000.0, kernel::MaternSmoothness::NU_1_5, ext_data_noise_stdv_ub));

    gaussian_process::kernel::MaternExtended kernel(hyperparams, hyperparams_lower, hyperparams_upper, conditional_indices, ext_data_conditional_indices);
    
    auto params_to_idx = kernel.GetOptParamsToIdx();
    auto ext_data_conditional_index_map = kernel.GetExtDataConditionalIndexMap();

}

TEST_F(GaussianProcessTestFixture, MaternKernelExtendedOptimizationTest) {
    const std::vector<double> ext_data_noise_stdv_lb = {0.01, 0.01, 0.01, 0.01};
    const std::vector<double> ext_data_noise_stdv_ub = {1000.0, 1000.0, 1000.0, 1000.0};

    auto hyperparams = std::make_shared<kernel::MaternExtendedHyperparams>(kernel::MaternExtendedHyperparams(0.5, 0.3, 0.5, kernel::MaternSmoothness::NU_1_5, ext_data_noise_stdv));
    auto hyperparams_lower = std::make_shared<kernel::MaternExtendedHyperparams>(kernel::MaternExtendedHyperparams(0.01, 0.01, 0.01, kernel::MaternSmoothness::NU_1_5, ext_data_noise_stdv_lb));
    auto hyperparams_upper = std::make_shared<kernel::MaternExtendedHyperparams>(kernel::MaternExtendedHyperparams(1000.0, 1000.0, 1000.0, kernel::MaternSmoothness::NU_1_5, ext_data_noise_stdv_ub));

    gaussian_process::kernel::MaternExtended kernel(hyperparams, hyperparams_lower, hyperparams_upper, conditional_indices, ext_data_conditional_indices);
    gaussian_process::GaussianProcess gp(x, y, conditional_indices, kernel);

    auto init_log_marginal_likelihood = gp.LogMarginalLikelihood();
    EXPECT_NO_THROW(gp.OptimizeHyperparameters(false));
    EXPECT_NO_THROW(gp.LogMarginalLikelihood());
    EXPECT_NO_THROW(gp.GetPosteriorGP());
    EXPECT_GT(gp.LogMarginalLikelihood(), init_log_marginal_likelihood) << "Log marginal likelihood did not improve after optimization";
}