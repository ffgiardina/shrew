#define _USE_MATH_DEFINES
#include <math.h>
#include "normal_distribution.h"
#include "arithmetic.h"
#include <cmath>
#include <complex>

namespace shrew {
namespace random_variable {

double NormalDistribution::Pdf(double x) 
{
    return 1.0 / (sigma * sqrt(2 * M_PI)) * exp(- pow(x - mu, 2) / (2 * pow(sigma, 2)));
};

double NormalDistribution::Cdf(double x) 
{
    return 1.0 / 2.0 * (1 + erf((x - mu) / (sigma * sqrt(2))));
};

double NormalDistribution::Mgf(double t) 
{
    return exp(mu * t + pow(sigma, 2) * pow(t, 2) / 2);
};

std::complex<double> NormalDistribution::Cf(double t) 
{
    std::complex<double> c(exp(-1.0/2.0 * pow(sigma * t, 2)) * cos(mu * t), 
        exp(-1.0/2.0 * pow(sigma * t, 2)) * sin(mu * t));
    return c;
};

RandomVariable<NormalDistribution> operator+(RandomVariable<NormalDistribution> const &var_a, RandomVariable<NormalDistribution> const &var_b)
{
    double mu = var_a.probability_distribution.mu + var_b.probability_distribution.mu;
    double sigma = sqrt(pow(var_a.probability_distribution.sigma, 2) + pow(var_b.probability_distribution.sigma, 2));
    return RandomVariable<NormalDistribution>(NormalDistribution(mu, sigma));
};

RandomVariable<NormalDistribution> operator-(RandomVariable<NormalDistribution> const &var_a, RandomVariable<NormalDistribution> const &var_b)
{
    double mu = var_a.probability_distribution.mu - var_b.probability_distribution.mu;
    double sigma = sqrt(pow(var_a.probability_distribution.sigma, 2) + pow(var_b.probability_distribution.sigma, 2));
    return RandomVariable<NormalDistribution>(NormalDistribution(mu, sigma));
};

RandomVariable<NormalDistribution> operator+(double var_a, RandomVariable<NormalDistribution> const &var_b)
{
    return RandomVariable<NormalDistribution>(NormalDistribution(var_a + var_b.probability_distribution.mu, var_b.probability_distribution.sigma));
};

RandomVariable<NormalDistribution> operator+(RandomVariable<NormalDistribution> const &var_a, double var_b)
{
    return RandomVariable<NormalDistribution>(NormalDistribution(var_b + var_a.probability_distribution.mu, var_a.probability_distribution.sigma));
};
RandomVariable<NormalDistribution> operator-(double var_a, RandomVariable<NormalDistribution> const &var_b)
{
    return RandomVariable<NormalDistribution>(NormalDistribution(var_a - var_b.probability_distribution.mu, var_b.probability_distribution.sigma));
};
RandomVariable<NormalDistribution> operator-(RandomVariable<NormalDistribution> const &var_a, double var_b)
{
    return RandomVariable<NormalDistribution>(NormalDistribution(var_a.probability_distribution.mu - var_b, var_a.probability_distribution.sigma));
};

RandomVariable<NormalDistribution> operator*(double var_a, RandomVariable<NormalDistribution> const &var_b)
{
    double mu = var_a * var_b.probability_distribution.mu;
    double sigma = var_a * var_b.probability_distribution.sigma;
    return RandomVariable<NormalDistribution>(NormalDistribution(mu, sigma));
};

RandomVariable<NormalDistribution> operator*(RandomVariable<NormalDistribution> const &var_a, double var_b)
{
    double mu = var_a.probability_distribution.mu * var_b;
    double sigma = var_a.probability_distribution.sigma * var_b;
    return RandomVariable<NormalDistribution>(NormalDistribution(mu, sigma));
};

RandomVariable<CompoundDistribution<double, NormalDistribution>> operator/(double var_a, RandomVariable<NormalDistribution> const &var_b)
{
    return RandomVariable<CompoundDistribution<double, NormalDistribution>>(CompoundDistribution<double, NormalDistribution>(var_a, std::make_shared<NormalDistribution>(var_b.probability_distribution), arithmetic::division));
};

RandomVariable<NormalDistribution> operator/(RandomVariable<NormalDistribution> const &var_a, double var_b)
{
    double mu = var_a.probability_distribution.mu / var_b;
    double sigma = var_a.probability_distribution.sigma / var_b;
    return RandomVariable<NormalDistribution>(NormalDistribution(mu, sigma));
};

RandomVariable<CompoundDistribution<NormalDistribution, NormalDistribution>> operator*(RandomVariable<NormalDistribution> const &var_a, RandomVariable<NormalDistribution> const &var_b)
{
    return RandomVariable<CompoundDistribution<NormalDistribution, NormalDistribution>>(CompoundDistribution<NormalDistribution, NormalDistribution>(std::make_shared<NormalDistribution>(var_a.probability_distribution), std::make_shared<NormalDistribution>(var_b.probability_distribution), arithmetic::multiplication));
};

RandomVariable<CompoundDistribution<NormalDistribution, NormalDistribution>> operator/(RandomVariable<NormalDistribution> const &var_a, RandomVariable<NormalDistribution> const &var_b)
{
    return RandomVariable<CompoundDistribution<NormalDistribution, NormalDistribution>>(CompoundDistribution<NormalDistribution, NormalDistribution>(std::make_shared<NormalDistribution>(var_a.probability_distribution), std::make_shared<NormalDistribution>(var_b.probability_distribution), arithmetic::division));
};

RandomVariable<CompoundDistribution<double, NormalDistribution>> operator^(double var_a, RandomVariable<NormalDistribution> const &var_b)
{
    return RandomVariable<CompoundDistribution<double, NormalDistribution>>(CompoundDistribution<double, NormalDistribution>(var_a, std::make_shared<NormalDistribution>(var_b.probability_distribution), arithmetic::exponentiation));
};

}  // namespace random_variable
}  // namespace shrew