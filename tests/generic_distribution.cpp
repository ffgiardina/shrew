#include <gtest/gtest.h>
#include "../src/ProbabilityDistribution/normal_distribution.h"
#include "../src/ProbabilityDistribution/compound_distribution.h"
#include "../src/ProbabilityDistribution/generic_distribution.h"

#define M_PI 3.14159265358979323846 /* pi */
using namespace shrew::random_variable;

class GenericDistributionTestFixture : public testing::Test
{
protected:
  std::function<double(double)> pdf = [](double x)
  { return 1 / sqrt(2 * M_PI) * exp(-pow(x, 2) / 2); };
  RandomVariable<GenericDistribution> generic = RandomVariable<GenericDistribution>(pdf);
};

TEST_F(GenericDistributionTestFixture, IntegrateCdfOfGenericVariable)
{
  ASSERT_NEAR(generic.probability_distribution.Cdf(INFINITY), 1.0, 1e-15);
}

TEST_F(GenericDistributionTestFixture, AdditionOfGenericVariables)
{
  ASSERT_NEAR((generic + generic).probability_distribution.Pdf(0.0), 0.282094791773878, 1e-15);
}
