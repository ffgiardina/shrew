#include "arithmetic.h"
#include "normal_distribution.h"

#define M_PI 3.14159265358979323846 /* pi */

namespace shrew
{
    namespace random_variable
    {

        double NormalDistribution::Pdf(double x) const
        {
            return 1.0 / (sigma * sqrt(2 * M_PI)) * exp(-pow(x - mu, 2) / (2 * pow(sigma, 2)));
        };

        double NormalDistribution::Cdf(double x) const
        {
            return 1.0 / 2.0 * (1 + erf((x - mu) / (sigma * sqrt(2))));
        };

        double NormalDistribution::Mgf(double t) const
        {
            return exp(mu * t + pow(sigma, 2) * pow(t, 2) / 2);
        };

        std::complex<double> NormalDistribution::Cf(double t) const
        {
            std::complex<double> c(exp(-1.0 / 2.0 * pow(sigma * t, 2)) * cos(mu * t),
                                   exp(-1.0 / 2.0 * pow(sigma * t, 2)) * sin(mu * t));
            return c;
        };

        std::tuple<const ProbabilityDistribution*, const ProbabilityDistribution*> NormalDistribution::get_operands() const 
        {
            return std::make_tuple(l_operand, r_operand);
        };

        RandomVariable<NormalDistribution> operator+(RandomVariable<NormalDistribution> const &var_a, RandomVariable<NormalDistribution> const &var_b)
        {
            
            double mu = var_a.probability_distribution.mu + var_b.probability_distribution.mu;
            double sigma = sqrt(pow(var_a.probability_distribution.sigma, 2) + pow(var_b.probability_distribution.sigma, 2));
            return RandomVariable<NormalDistribution>(NormalDistribution(mu, sigma, &(var_a.probability_distribution), &(var_b.probability_distribution)));
        };

        RandomVariable<NormalDistribution> operator-(RandomVariable<NormalDistribution> const &var_a, RandomVariable<NormalDistribution> const &var_b)
        {
            double mu = var_a.probability_distribution.mu - var_b.probability_distribution.mu;
            double sigma = sqrt(pow(var_a.probability_distribution.sigma, 2) + pow(var_b.probability_distribution.sigma, 2));
            return RandomVariable<NormalDistribution>(NormalDistribution(mu, sigma, &(var_a.probability_distribution), &(var_b.probability_distribution)));
        };

        RandomVariable<NormalDistribution> operator+(double var_a, RandomVariable<NormalDistribution> const &var_b)
        {
            return RandomVariable<NormalDistribution>(NormalDistribution(var_a + var_b.probability_distribution.mu, var_b.probability_distribution.sigma, 0, &(var_b.probability_distribution)));
        };

        RandomVariable<NormalDistribution> operator+(RandomVariable<NormalDistribution> const &var_a, double var_b)
        {
            return RandomVariable<NormalDistribution>(NormalDistribution(var_b + var_a.probability_distribution.mu, var_a.probability_distribution.sigma, &(var_a.probability_distribution), 0));
        };
        RandomVariable<NormalDistribution> operator-(double var_a, RandomVariable<NormalDistribution> const &var_b)
        {
            return RandomVariable<NormalDistribution>(NormalDistribution(var_a - var_b.probability_distribution.mu, var_b.probability_distribution.sigma, 0, &(var_b.probability_distribution)));
        };
        RandomVariable<NormalDistribution> operator-(RandomVariable<NormalDistribution> const &var_a, double var_b)
        {
            return RandomVariable<NormalDistribution>(NormalDistribution(var_a.probability_distribution.mu - var_b, var_a.probability_distribution.sigma, &(var_a.probability_distribution), 0));
        };

        RandomVariable<NormalDistribution> operator*(double var_a, RandomVariable<NormalDistribution> const &var_b)
        {
            double mu = var_a * var_b.probability_distribution.mu;
            double sigma = var_a * var_b.probability_distribution.sigma;
            return RandomVariable<NormalDistribution>(NormalDistribution(mu, sigma, 0, &(var_b.probability_distribution)));
        };

        RandomVariable<NormalDistribution> operator*(RandomVariable<NormalDistribution> const &var_a, double var_b)
        {
            double mu = var_a.probability_distribution.mu * var_b;
            double sigma = var_a.probability_distribution.sigma * var_b;
            return RandomVariable<NormalDistribution>(NormalDistribution(mu, sigma, &(var_a.probability_distribution), 0));
        };

        RandomVariable<NormalDistribution> operator/(RandomVariable<NormalDistribution> const &var_a, double var_b)
        {
            double mu = var_a.probability_distribution.mu / var_b;
            double sigma = var_a.probability_distribution.sigma / var_b;
            return RandomVariable<NormalDistribution>(NormalDistribution(mu, sigma, &(var_a.probability_distribution), 0));
        };

    } // namespace random_variable
} // namespace shrew