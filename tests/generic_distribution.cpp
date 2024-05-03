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
  ASSERT_NEAR(res, 1.0, 1e-15);
}

TEST_F(GenericDistributionTestFixture, MultiplicationOfTwoNormalRandomVariables) {
  RandomVariable<NormalDistribution> normal_a = RandomVariable<NormalDistribution>(NormalDistribution(-1.0, 1.0));
  RandomVariable<NormalDistribution> normal_b = RandomVariable<NormalDistribution>(NormalDistribution(1.0, 1.0));
  ASSERT_NEAR((normal_a * normal_b).probability_distribution.Pdf(1.0), 0.0759651, 1e-8);
}

TEST_F(GenericDistributionTestFixture, DivisionOfTwoNormalRandomVariablesResultsInCauchyDistribution) {
  RandomVariable<NormalDistribution> normal_a = RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));
  RandomVariable<NormalDistribution> normal_b = RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));
  ASSERT_NEAR((normal_a / normal_b).probability_distribution.Pdf(0.0), 0.318309886183790, 1e-15);
}

TEST_F(GenericDistributionTestFixture, LeftConstantExponentiation) {
  RandomVariable<NormalDistribution> normal_a = RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));
    
  ASSERT_NEAR((2.0 ^ normal_a).probability_distribution.Pdf(1.0), 0.575, 1e-3);
  ASSERT_THROW(((-2.0) ^ normal_a).probability_distribution.Pdf(1.0), std::logic_error);
}

TEST_F(GenericDistributionTestFixture, AdditionOfTwoNormalRandomVariablesAsGenerics) {
  RandomVariable<NormalDistribution> normal_a = RandomVariable<NormalDistribution>(NormalDistribution(1.0, sqrt(2.0)));
  RandomVariable<NormalDistribution> normal_b = RandomVariable<NormalDistribution>(NormalDistribution(-1.0, sqrt(2.0)));
  
  GenericDistribution g_dist_a = GenericDistribution(std::make_shared<NormalDistribution>(NormalDistribution(1.0, 1.0)), std::make_shared<NormalDistribution>(NormalDistribution(0.0, 1.0)), arithmetic::Operation::addition);
  GenericDistribution g_dist_b = GenericDistribution(std::make_shared<NormalDistribution>(NormalDistribution(-1.0, 1.0)), std::make_shared<NormalDistribution>(NormalDistribution(0.0, 1.0)), arithmetic::Operation::addition);
  
  using GNN = GenericDistribution<NormalDistribution, NormalDistribution>;
  RandomVariable<GNN> gen_a = RandomVariable<GNN>(g_dist_a);
  RandomVariable<GNN> gen_b = RandomVariable<GNN>(g_dist_b);

  ASSERT_NEAR((gen_a + gen_b).probability_distribution.Pdf(0.0), (normal_a + normal_b).probability_distribution.Pdf(0.0), 1e-15);
}

TEST_F(GenericDistributionTestFixture, SubtractionOfTwoNormalRandomVariablesAsGenerics) {
  RandomVariable<NormalDistribution> normal_a = RandomVariable<NormalDistribution>(NormalDistribution(1.0, sqrt(2.0)));
  RandomVariable<NormalDistribution> normal_b = RandomVariable<NormalDistribution>(NormalDistribution(-1.0, sqrt(2.0)));

  GenericDistribution g_dist_a = GenericDistribution(std::make_shared<NormalDistribution>(NormalDistribution(1.0, 1.0)), std::make_shared<NormalDistribution>(NormalDistribution(0.0, 1.0)), arithmetic::Operation::addition);
  GenericDistribution g_dist_b = GenericDistribution(std::make_shared<NormalDistribution>(NormalDistribution(-1.0, 1.0)), std::make_shared<NormalDistribution>(NormalDistribution(0.0, 1.0)), arithmetic::Operation::addition);
  
  using GNN = GenericDistribution<NormalDistribution, NormalDistribution>;
  RandomVariable<GNN> gen_a = RandomVariable<GNN>(g_dist_a);
  RandomVariable<GNN> gen_b = RandomVariable<GNN>(g_dist_b);

  ASSERT_NEAR((gen_a - gen_b).probability_distribution.Pdf(0.0), (normal_a - normal_b).probability_distribution.Pdf(0.0), 1e-15);
  ASSERT_NEAR((gen_b - gen_a).probability_distribution.Pdf(0.0), (normal_b - normal_a).probability_distribution.Pdf(0.0), 1e-15);
}

// TEST_F(GenericDistributionTestFixture, ExponentiationOfTwoNormalRandomVariables) {
//   RandomVariable<NormalDistribution> normal_a = RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));
//   RandomVariable<NormalDistribution> normal_b = RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));
  
//   double res = (normal_a ^ normal_b).probability_distribution.Pdf(2.0);
//   ASSERT_NEAR((normal_a ^ normal_b).probability_distribution.Pdf(1.0), 0.0759651, 1e-8);
// }