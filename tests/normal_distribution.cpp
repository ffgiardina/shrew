#include <gtest/gtest.h>
#include "../src/normal_distribution.h"

using namespace shrew::random_variable;

// Demonstrate some basic assertions.
TEST(NormalDistributionUnitTest, ProbabilityDensityFunction) {
  double sigma = 1.0;
  double mu = 0.0;
  
  NormalDistribution normal = NormalDistribution(mu, sigma);
  RandomVariable<NormalDistribution> grv = RandomVariable<NormalDistribution>(normal);
  
  ASSERT_NEAR(grv.Evaluate(0), 0.3989422804014337, 1e-15);
}