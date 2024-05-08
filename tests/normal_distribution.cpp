#include <gtest/gtest.h>
#include "../src/ProbabilityDistribution/normal_distribution.h"

using namespace shrew::random_variable;

class NormalDistributionTestFixture : public testing::Test
{
protected:
  NormalDistributionTestFixture() {}
  RandomVariable<NormalDistribution> grv = RandomVariable<NormalDistribution>(NormalDistribution(0, 1.0));
};

TEST_F(NormalDistributionTestFixture, ProbabilityDensityFunction)
{
  ASSERT_NEAR(grv.probability_distribution.Pdf(0), 0.3989422804014337, 1e-15);
}

TEST_F(NormalDistributionTestFixture, CumulativeDistributionFunction)
{
  ASSERT_NEAR(grv.probability_distribution.Cdf(0), 0.5, 1e-15);
}

TEST_F(NormalDistributionTestFixture, MomentGeneratingFunction)
{
  ASSERT_NEAR(grv.probability_distribution.Mgf(1.0), 1.6487212707001281, 1e-15);
}

TEST_F(NormalDistributionTestFixture, CharacteristicFunction)
{
  std::complex<double> c_ref(0.60653065971263342, 0);
  double diff = abs(grv.probability_distribution.Cf(1.0) - c_ref);
  ASSERT_NEAR(diff, 0, 1e-15);
}

TEST_F(NormalDistributionTestFixture, RandomVariableAddition)
{
  auto grv_a = RandomVariable<NormalDistribution>(NormalDistribution(3.0, 1.0));
  ASSERT_NEAR((grv + grv).probability_distribution.Pdf(0), 1 / sqrt(2) * 0.3989422804014337, 1e-15);
  ASSERT_NEAR((grv_a + grv).probability_distribution.Pdf(0), 0.029732572305907347, 1e-15);
  ASSERT_NEAR((grv + grv_a).probability_distribution.Pdf(0), 0.029732572305907347, 1e-15);
}

TEST_F(NormalDistributionTestFixture, RandomVariableSubtraction)
{
  auto grv_a = RandomVariable<NormalDistribution>(NormalDistribution(3.0, 1.0));
  ASSERT_NEAR((grv - grv).probability_distribution.Pdf(0), 1 / sqrt(2) * 0.3989422804014337, 1e-15);
  ASSERT_NEAR((grv_a - grv).probability_distribution.Pdf(3.0), 1 / sqrt(2) * 0.3989422804014337, 1e-15);
  ASSERT_NEAR((grv - grv_a).probability_distribution.Pdf(-3.0), 1 / sqrt(2) * 0.3989422804014337, 1e-15);
}

TEST_F(NormalDistributionTestFixture, ConstantAddition)
{
  ASSERT_NEAR((grv + 2).probability_distribution.Pdf(2.0), 0.3989422804014337, 1e-15);
  ASSERT_NEAR((2 + grv).probability_distribution.Pdf(2.0), 0.3989422804014337, 1e-15);
}

TEST_F(NormalDistributionTestFixture, ConstantSubtraction)
{
  ASSERT_NEAR((grv - 2).probability_distribution.Pdf(-2.0), 0.3989422804014337, 1e-15);
  ASSERT_NEAR((2 - grv).probability_distribution.Pdf(2.0), 0.3989422804014337, 1e-15);
}

TEST_F(NormalDistributionTestFixture, ConstantMultiplication)
{
  ASSERT_NEAR((grv * 2).probability_distribution.Pdf(0.0), 0.3989422804014337 / 2.0, 1e-15);
  ASSERT_NEAR((2 * grv).probability_distribution.Pdf(0.0), 0.3989422804014337 / 2.0, 1e-15);
}

TEST_F(NormalDistributionTestFixture, ConstantDivision)
{
  ASSERT_NEAR((grv / 2).probability_distribution.Pdf(0.0), 0.3989422804014337 * 2.0, 1e-14);
  ASSERT_NEAR((1 / grv).probability_distribution.Pdf(1.0), 0.2419707245191433, 1e-14);
}