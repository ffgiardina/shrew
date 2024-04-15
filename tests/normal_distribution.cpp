#include <gtest/gtest.h>
#include "../src/normal_distribution.h"

using namespace shrew::random_variable;

class NormalDistributionTestFixture : public testing::Test {
 protected:
  NormalDistributionTestFixture() {}
  RandomVariable<NormalDistribution> grv = RandomVariable<NormalDistribution>(NormalDistribution(0, 1.0));
};

TEST_F(NormalDistributionTestFixture, ProbabilityDensityFunction) {
  ASSERT_NEAR(grv.probability_distribution.Pdf(0), 0.3989422804014337, 1e-15);
}

TEST_F(NormalDistributionTestFixture, CumulativeDistributionFunction) {
  ASSERT_NEAR(grv.probability_distribution.Cdf(0), 0.5, 1e-15);
}

TEST_F(NormalDistributionTestFixture, MomentGeneratingFunction) {
  ASSERT_NEAR(grv.probability_distribution.Mgf(1.0), 1.6487212707001281, 1e-15);
}

TEST_F(NormalDistributionTestFixture, CharacteristicFunction) {
  std::complex<double> c_ref(0.60653065971263342, 0);
  double diff = abs(grv.probability_distribution.Cf(1.0) - c_ref);
  ASSERT_NEAR(diff, 0, 1e-15);
}

TEST_F(NormalDistributionTestFixture, RandomVariableAddition) {
  RandomVariable<NormalDistribution> grv_a = RandomVariable<NormalDistribution>(NormalDistribution(3.0, 1.0));
  ASSERT_NEAR((grv + grv).probability_distribution.Pdf(0), 0.5*0.3989422804014337, 1e-15);
  ASSERT_NEAR((grv_a + grv).probability_distribution.Pdf(0), 0.06475879783294586, 1e-15);
  ASSERT_NEAR((grv + grv_a).probability_distribution.Pdf(0), 0.06475879783294586, 1e-15);
}

TEST_F(NormalDistributionTestFixture, RandomVariableSubtraction) {
  RandomVariable<NormalDistribution> grv_a = RandomVariable<NormalDistribution>(NormalDistribution(3.0, 1.0));

  ASSERT_NEAR((grv - grv).probability_distribution.Pdf(0), 0.5*0.3989422804014337, 1e-15);
  ASSERT_NEAR((grv_a - grv).probability_distribution.Pdf(3.0), 0.5*0.3989422804014337, 1e-15);
  ASSERT_NEAR((grv - grv_a).probability_distribution.Pdf(-3.0), 0.5*0.3989422804014337, 1e-15);
}