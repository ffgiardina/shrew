#include "multivariate_normal.h"

#include <iostream>
#include <gtest/gtest.h>

#define M_PI 3.14159265358979323846 /* pi */
using namespace shrew::random_vector;

class MultivariateNormalTestFixture : public testing::Test {
 protected:
};

TEST_F(MultivariateNormalTestFixture, InitializationTest) {
  auto mu = Eigen::Matrix<double, 2, 1> {0, 0};
  auto K = Eigen::Matrix<double, 2, 2> {{1, 0}, {0, 1}};
  ASSERT_NEAR(MultivariateNormal<2>(mu, K).joint_pdf(Eigen::Vector2d(0, 0)), 1.0 / (2 * M_PI), 1e-15);
}

TEST_F(MultivariateNormalTestFixture, BivariateNormalTest) {
  auto mu = Eigen::Matrix<double, 2, 1> {0.5, -0.2};
  auto K = Eigen::Matrix<double, 2, 2> {{2.0, 0.3}, {0.3, 0.5}};

  ASSERT_NEAR(MultivariateNormal<2>(mu, K).joint_pdf(Eigen::Vector2d(1.0, 1.0)), 0.03900683351100838, 1e-15);
}

TEST_F(MultivariateNormalTestFixture, TrivariateNormalTest) {
  auto mu = Eigen::Matrix<double, 3, 1> {0.5, -0.2, 0.1};
  auto K = Eigen::Matrix<double, 3, 3> {{2.0, 0.3, 0.5}, {0.3, 1.0, 0.6}, {0.5, 0.6, 1.3}};

  ASSERT_NEAR(MultivariateNormal<3>(mu, K).joint_pdf(Eigen::Vector3d(1.0, 1.0, -1.0)), 0.0034750315971456835, 1e-15);
}

TEST_F(MultivariateNormalTestFixture, MarginalNormalTest) {
  auto mu = Eigen::Matrix<double, 3, 1> {0.5, -0.2, 0.1};
  auto K = Eigen::Matrix<double, 3, 3> {{2.0, 0.3, 0.5}, {0.3, 1.0, 0.6}, {0.5, 0.6, 1.3}};
  auto mvec = MultivariateNormal<3>(mu, K);
  auto marginal = getMarginal<3,2>(MultivariateNormal<3>(mu, K), std::vector<int>{ 0, 2 });

  auto K_confirm = Eigen::Matrix<double, 2, 2> {{2.0, 0.5}, {0.5, 1.3}};
  auto mu_confirm = Eigen::Matrix<double, 2, 1> {0.5, 0.1};
  auto x_confirm = Eigen::Vector2d(1.0, -1.0);
  ASSERT_EQ(marginal.joint_pdf(x_confirm), MultivariateNormal<2>(mu_confirm, K_confirm).joint_pdf(x_confirm));
}