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
