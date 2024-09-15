#include "shrew/generic_distribution.hpp"

#include <gtest/gtest.h>

#include "shrew/compound_distribution.hpp"
#include "shrew/normal_distribution.hpp"

using namespace shrew::random_variable;

class GenericDistributionTestFixture : public testing::Test
{
protected:
  std::function<double(double)> pdf = [](double x)
  {
    return 1 / sqrt(2 * M_PI) * exp(-pow(x, 2) / 2);
  };
  RandomVariable<GenericDistribution> generic_a =
      RandomVariable<GenericDistribution>(pdf);
  RandomVariable<GenericDistribution> generic_b =
      RandomVariable<GenericDistribution>(pdf);
};

TEST_F(GenericDistributionTestFixture, IntegrateCdfOfGenericVariable)
{
  ASSERT_NEAR(generic_a.probability_distribution.Cdf(INFINITY), 1.0, 1e-15);
}

TEST_F(GenericDistributionTestFixture, AdditionOfGenericVariables)
{
  ASSERT_NEAR((generic_a + generic_b).probability_distribution.Pdf(0.0),
              0.282094791773878, 1e-15);
}
