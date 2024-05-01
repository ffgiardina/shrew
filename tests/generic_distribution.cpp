#include <gtest/gtest.h>
#include "../src/normal_distribution.h"
#include "../src/generic_distribution.h"
#include "../src/numerical_methods.h"

using namespace shrew::random_variable;

class GenericDistributionTestFixture : public testing::Test {
 protected:
  GenericDistributionTestFixture() {}
};

TEST_F(GenericDistributionTestFixture, IntegrationVerification) {
  shrew::numerical_methods::InfiniteDomainGaussKronrod integrator = shrew::numerical_methods::InfiniteDomainGaussKronrod();
  RandomVariable<NormalDistribution> normal_a = RandomVariable<NormalDistribution>(NormalDistribution(0, 1.0));

  double res = integrator.Integrate([&normal_a](double x){return normal_a.probability_distribution.Pdf(x);});
  ASSERT_NEAR(1.0, 1.0, 1e-15);
}

TEST_F(GenericDistributionTestFixture, MultiplicationOfTwoNormalRandomVariables) {
  RandomVariable<NormalDistribution> normal_a = RandomVariable<NormalDistribution>(NormalDistribution(-1.0, 1.0));
  RandomVariable<NormalDistribution> normal_b = RandomVariable<NormalDistribution>(NormalDistribution(1.0, 1.0));
  double res = (normal_a * normal_b).probability_distribution.Pdf(1.0);
  ASSERT_NEAR((normal_a * normal_b).probability_distribution.Pdf(1.0), 0.0759651, 1e-8);
}
