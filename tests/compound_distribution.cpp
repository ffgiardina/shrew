#include "compound_distribution.h"
#include "normal_distribution.h"
#include "../src/numerical_methods.h"

#include <gtest/gtest.h>

using namespace shrew::random_variable;

class CompoundDistributionTestFixture : public testing::Test
{
protected:
  CompoundDistributionTestFixture() {}
};

TEST_F(CompoundDistributionTestFixture, IntegrationVerification)
{
  const shrew::numerical_methods::Integrator &integrator = shrew::numerical_methods::GaussKronrod();
  RandomVariable<NormalDistribution> normal_a = RandomVariable<NormalDistribution>(NormalDistribution(0, 1.0));

  double res = integrator.Integrate([&normal_a](double x)
                                    { return normal_a.probability_distribution.Pdf(x); });
  ASSERT_NEAR(res, 1.0, 1e-15);
}

TEST_F(CompoundDistributionTestFixture, MultiplicationOfTwoNormalRandomVariables)
{
  RandomVariable<NormalDistribution> normal_a = RandomVariable<NormalDistribution>(NormalDistribution(-1.0, 1.0));
  RandomVariable<NormalDistribution> normal_b = RandomVariable<NormalDistribution>(NormalDistribution(1.0, 1.0));
  ASSERT_NEAR((normal_a * normal_b).probability_distribution.Pdf(1.0), 0.0759651, 1e-8);
}

TEST_F(CompoundDistributionTestFixture, DivisionOfTwoNormalRandomVariablesResultsInCauchyDistribution)
{
  RandomVariable<NormalDistribution> normal_a = RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));
  RandomVariable<NormalDistribution> normal_b = RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));
  ASSERT_NEAR((normal_a / normal_b).probability_distribution.Pdf(0.0), 0.318309886183790, 1e-15);
}

TEST_F(CompoundDistributionTestFixture, LeftConstantExponentiation)
{
  RandomVariable<NormalDistribution> normal_a = RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));

  ASSERT_NEAR((2.0 ^ normal_a).probability_distribution.Pdf(1.0), 0.575, 1e-3);
  ASSERT_THROW(((-2.0) ^ normal_a).probability_distribution.Pdf(1.0), std::logic_error);
}

TEST_F(CompoundDistributionTestFixture, AdditionOfTwoNormalRandomVariablesAsGenerics)
{
  RandomVariable<NormalDistribution> normal_a = RandomVariable<NormalDistribution>(NormalDistribution(1.0, sqrt(2.0)));
  RandomVariable<NormalDistribution> normal_b = RandomVariable<NormalDistribution>(NormalDistribution(-1.0, sqrt(2.0)));

  CompoundDistribution g_dist_a = CompoundDistribution(std::make_shared<NormalDistribution>(NormalDistribution(1.0, 1.0)), std::make_shared<NormalDistribution>(NormalDistribution(0.0, 1.0)), arithmetic::Operation::addition);
  CompoundDistribution g_dist_b = CompoundDistribution(std::make_shared<NormalDistribution>(NormalDistribution(-1.0, 1.0)), std::make_shared<NormalDistribution>(NormalDistribution(0.0, 1.0)), arithmetic::Operation::addition);

  using GNN = CompoundDistribution<NormalDistribution, NormalDistribution>;
  RandomVariable<GNN> gen_a = RandomVariable<GNN>(g_dist_a);
  RandomVariable<GNN> gen_b = RandomVariable<GNN>(g_dist_b);

  ASSERT_NEAR((gen_a + gen_b).probability_distribution.Pdf(0.0), (normal_a + normal_b).probability_distribution.Pdf(0.0), 1e-15);
}

TEST_F(CompoundDistributionTestFixture, SubtractionOfTwoNormalRandomVariablesAsGenerics)
{
  RandomVariable<NormalDistribution> normal_a = RandomVariable<NormalDistribution>(NormalDistribution(1.0, sqrt(2.0)));
  RandomVariable<NormalDistribution> normal_b = RandomVariable<NormalDistribution>(NormalDistribution(-1.0, sqrt(2.0)));

  CompoundDistribution g_dist_a = CompoundDistribution(std::make_shared<NormalDistribution>(NormalDistribution(1.0, 1.0)), std::make_shared<NormalDistribution>(NormalDistribution(0.0, 1.0)), arithmetic::Operation::addition);
  CompoundDistribution g_dist_b = CompoundDistribution(std::make_shared<NormalDistribution>(NormalDistribution(-1.0, 1.0)), std::make_shared<NormalDistribution>(NormalDistribution(0.0, 1.0)), arithmetic::Operation::addition);

  using GNN = CompoundDistribution<NormalDistribution, NormalDistribution>;
  RandomVariable<GNN> gen_a = RandomVariable<GNN>(g_dist_a);
  RandomVariable<GNN> gen_b = RandomVariable<GNN>(g_dist_b);

  ASSERT_NEAR((gen_a - gen_b).probability_distribution.Pdf(0.0), (normal_a - normal_b).probability_distribution.Pdf(0.0), 1e-15);
  ASSERT_NEAR((gen_b - gen_a).probability_distribution.Pdf(0.0), (normal_b - normal_a).probability_distribution.Pdf(0.0), 1e-15);
}

TEST_F(CompoundDistributionTestFixture, ExponentiationOfTwoNormalRandomVariables)
{
  RandomVariable<NormalDistribution> normal_a = RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));
  RandomVariable<NormalDistribution> normal_b = RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));

  ASSERT_THROW((normal_a ^ normal_b).probability_distribution.Pdf(1.0), std::logic_error);
}

TEST_F(CompoundDistributionTestFixture, CDFOfStandardNormalRandomVariables)
{
  RandomVariable<NormalDistribution> normal_a = RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));
  RandomVariable<NormalDistribution> normal_b = RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));

  auto generic_rv = normal_a * normal_b;
  auto a = generic_rv.probability_distribution.Pdf(0.0);
  ASSERT_NEAR(generic_rv.probability_distribution.Cdf(0.0), 0.5, 1e-6);
  ASSERT_NEAR(generic_rv.probability_distribution.Cdf(INFINITY), 1.0, 1e-6);
}