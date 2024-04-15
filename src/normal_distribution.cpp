#include <math.h>
#include "normal_distribution.h"
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

template<>
RandomVariable<NormalDistribution> RandomVariable<NormalDistribution>::operator+(RandomVariable<NormalDistribution> const &var) 
{
    double mu = probability_distribution.mu + var.probability_distribution.mu;
    double sigma = pow(probability_distribution.sigma, 2) + pow(var.probability_distribution.sigma, 2);
    return RandomVariable<NormalDistribution>(NormalDistribution(mu, sigma));
};

template<>
RandomVariable<NormalDistribution> RandomVariable<NormalDistribution>::operator-(RandomVariable<NormalDistribution> const &var) 
{
    double mu = probability_distribution.mu - var.probability_distribution.mu;
    double sigma = pow(probability_distribution.sigma, 2) + pow(var.probability_distribution.sigma, 2);
    return RandomVariable<NormalDistribution>(NormalDistribution(mu, sigma));
};

}  // namespace random_variable
}  // namespace shrew