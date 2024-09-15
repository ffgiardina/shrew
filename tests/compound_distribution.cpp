#include "shrew/compound_distribution.hpp"

#include <gtest/gtest.h>
#include <math.h>

#include <shrew/numerical_methods.hpp>
#include "shrew/normal_distribution.hpp"

using namespace shrew::random_variable;

class CompoundDistributionTestFixture : public testing::Test
{
};

TEST_F(CompoundDistributionTestFixture, IntegrationVerification)
{
    const shrew::numerical_methods::Integrator &integrator =
        shrew::numerical_methods::GaussKronrod();
    RandomVariable<NormalDistribution> normal_a =
        RandomVariable<NormalDistribution>(NormalDistribution(0, 1.0));

    double res = integrator.Integrate([&normal_a](double x)
                                      { return normal_a.probability_distribution.Pdf(x); });
    ASSERT_NEAR(res, 1.0, 1e-15);
}

TEST_F(CompoundDistributionTestFixture,
       MultiplicationOfSelfReferencingNormalRandomVariables)
{
    RandomVariable<NormalDistribution> normal_a =
        RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));
    RandomVariable<NormalDistribution> normal_b =
        RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));
    RandomVariable<NormalDistribution> normal_c =
        RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));
    ASSERT_THROW((normal_a * normal_a).probability_distribution.Pdf(0.0),
                 std::logic_error);
    ASSERT_THROW((normal_a + 2 + normal_a).probability_distribution.Pdf(0.0),
                 std::logic_error);
    ASSERT_THROW((normal_a + 2 + normal_b * normal_c - 3 * 5 * normal_a)
                     .probability_distribution.Pdf(0.0),
                 std::logic_error);
}

TEST_F(CompoundDistributionTestFixture,
       MultiplicationOfTwoNormalRandomVariables)
{
    RandomVariable<NormalDistribution> normal_a =
        RandomVariable<NormalDistribution>(NormalDistribution(-1.0, 1.0));
    RandomVariable<NormalDistribution> normal_b =
        RandomVariable<NormalDistribution>(NormalDistribution(1.0, 1.0));
    ASSERT_NEAR((normal_a * normal_b).probability_distribution.Pdf(1.0),
                0.0759651, 1e-8);

    RandomVariable<NormalDistribution> normal_s1 =
        RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));
    RandomVariable<NormalDistribution> normal_s2 =
        RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));
    ASSERT_NEAR((normal_s1 * normal_s2).probability_distribution.Pdf(1.0),
                std::cyl_bessel_k(0, 1.0) / M_PI, 1e-15);
}

TEST_F(CompoundDistributionTestFixture,
       ConstantOperationOnCompoundDistribution)
{
    RandomVariable<NormalDistribution> normal_a =
        RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));
    RandomVariable<NormalDistribution> normal_b =
        RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));
    ASSERT_NEAR((2 + normal_a * normal_b).probability_distribution.Pdf(3.0),
                std::cyl_bessel_k(0, 1.0) / M_PI, 1e-15);
    ASSERT_NEAR((normal_a * normal_b - 2.0).probability_distribution.Pdf(-1.0),
                std::cyl_bessel_k(0, 1.0) / M_PI, 1e-15);
}

TEST_F(CompoundDistributionTestFixture,
       MultiplicationOfThreeNormalRandomVariables)
{
    RandomVariable<NormalDistribution> normal_a =
        RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));
    RandomVariable<NormalDistribution> normal_b =
        RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));
    RandomVariable<NormalDistribution> normal_c =
        RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));

    double meijer_g = 1.362710108013438627;
    ASSERT_NEAR(
        (normal_a * normal_b * normal_c).probability_distribution.Pdf(1.0),
        meijer_g / (2 * sqrt(2) * pow(M_PI, 1.5)), 1e-15);
}

TEST_F(CompoundDistributionTestFixture,
       DivisionOfTwoNormalRandomVariablesResultsInCauchyDistribution)
{
    RandomVariable<NormalDistribution> normal_a =
        RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));
    RandomVariable<NormalDistribution> normal_b =
        RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));
    ASSERT_NEAR((normal_a / normal_b).probability_distribution.Pdf(0.0),
                0.318309886183790, 1e-15);
}

TEST_F(CompoundDistributionTestFixture, LeftConstantExponentiation)
{
    RandomVariable<NormalDistribution> normal_a =
        RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));

    ASSERT_NEAR((2.0 ^ normal_a).probability_distribution.Pdf(1.0),
                1 / (log(2) * sqrt(2 * M_PI)), 1e-15);
    ASSERT_NEAR((3.0 ^ normal_a).probability_distribution.Pdf(1.0),
                1 / (log(3) * sqrt(2 * M_PI)), 1e-15);
    ASSERT_THROW((2.0 ^ normal_a).probability_distribution.Pdf(-1.0),
                 std::logic_error);
    ASSERT_THROW(((-2.0) ^ normal_a).probability_distribution.Pdf(1.0),
                 std::logic_error);
}

TEST_F(CompoundDistributionTestFixture, RightConstantExponentiation)
{
    RandomVariable<NormalDistribution> normal_a =
        RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));
    RandomVariable<NormalDistribution> normal_b =
        RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));

    ASSERT_NEAR((normal_a ^ 2.0).probability_distribution.Pdf(1.0),
                1.0 / (sqrt(2.0 * M_PI)) * exp(-1.0 / 2.0), 1e-15);
    ASSERT_THROW((normal_a ^ 2.0).probability_distribution.Pdf(-1.0),
                 std::logic_error);
    ASSERT_NEAR((normal_a ^ 3.0).probability_distribution.Pdf(1.0),
                1.0 / (3.0 * sqrt(2.0 * M_PI)) * exp(-1.0 / 2.0), 1e-15);
    ASSERT_NEAR((normal_a ^ 3.0).probability_distribution.Pdf(-1.0),
                1.0 / (3.0 * sqrt(2.0 * M_PI)) * exp(-1.0 / 2.0), 1e-15);
    ASSERT_NEAR((normal_a ^ (-2.0)).probability_distribution.Pdf(1.0),
                1.0 / sqrt(2 * M_PI * exp(1)), 1e-15);
    ASSERT_THROW((normal_a ^ (-2.0)).probability_distribution.Pdf(-1.0),
                 std::logic_error);
    ASSERT_NEAR((normal_a ^ (-3.0)).probability_distribution.Pdf(1.0),
                1.0 / (3.0 * sqrt(2 * M_PI * exp(1))), 1e-15);
    ASSERT_NEAR((normal_a ^ (-3.0)).probability_distribution.Pdf(-1.0),
                1.0 / (3.0 * sqrt(2 * M_PI * exp(1))), 1e-15);
    ASSERT_THROW((normal_a ^ 0.3).probability_distribution.Pdf(1.0),
                 std::logic_error);
}

TEST_F(CompoundDistributionTestFixture,
       ExponentiationOfTwoNormalRandomVariables)
{
    RandomVariable<NormalDistribution> normal_a =
        RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));
    RandomVariable<NormalDistribution> normal_b =
        RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));

    ASSERT_THROW((normal_a ^ normal_b).probability_distribution.Pdf(1.0),
                 std::logic_error);
}

TEST_F(CompoundDistributionTestFixture, CDFOfStandardNormalRandomVariables)
{
    RandomVariable<NormalDistribution> normal_a =
        RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));
    RandomVariable<NormalDistribution> normal_b =
        RandomVariable<NormalDistribution>(NormalDistribution(0.0, 1.0));

    auto generic_rv = normal_a * normal_b;
    ASSERT_NEAR(generic_rv.probability_distribution.Cdf(0.0), 0.5, 1e-6);
    ASSERT_NEAR(generic_rv.probability_distribution.Cdf(INFINITY), 1.0, 1e-6);
}

TEST_F(CompoundDistributionTestFixture, CompoundOperations)
{
    RandomVariable<NormalDistribution> normal_a =
        RandomVariable<NormalDistribution>(NormalDistribution(-1.0, 1.0));
    RandomVariable<NormalDistribution> normal_b =
        RandomVariable<NormalDistribution>(NormalDistribution(-1.0, 1.0));
    RandomVariable<NormalDistribution> normal_c =
        RandomVariable<NormalDistribution>(NormalDistribution(1.0, 1.0));
    RandomVariable<NormalDistribution> normal_d =
        RandomVariable<NormalDistribution>(NormalDistribution(1.0, 1.0));
    auto normal_ac = normal_a * normal_c;
    auto normal_bd = normal_b * normal_d;
    ASSERT_NEAR((normal_a + normal_b + normal_c + normal_d)
                    .probability_distribution.Pdf(0.0),
                1 / sqrt(2 * M_PI * 4), 1e-15);
    ASSERT_NEAR((2 * normal_a + normal_c * 2).probability_distribution.Pdf(0.0),
                1 / sqrt(2 * M_PI * 8), 1e-15);
    ASSERT_NEAR(
        ((normal_a + normal_c) * (normal_b + normal_d))
            .probability_distribution.Pdf(0.0),
        (RandomVariable<NormalDistribution>(NormalDistribution(0.0, sqrt(2))) *
         RandomVariable<NormalDistribution>(NormalDistribution(0.0, sqrt(2))))
            .probability_distribution.Pdf(0.0),
        1e-15);
    ASSERT_NEAR((normal_a * normal_c + normal_b * normal_d)
                    .probability_distribution.Pdf(0.0),
                (normal_ac + normal_bd).probability_distribution.Pdf(0.0), 1e-15);
}