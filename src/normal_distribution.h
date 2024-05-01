#pragma once

#include "random_variable.h"
#include "generic_distribution.h"
#include <complex>

namespace shrew {
namespace random_variable {

/// @brief Normal (or Gaussian) probability distribution 
class NormalDistribution : public ProbabilityDistribution {
 public:
  // Probability density function
  virtual double Pdf(double x) override;

  // Cumulative distribution function
  virtual double Cdf(double x) override;

  // Moment generating function
  virtual double Mgf(double x) override;

  // Characteristic function
  virtual std::complex<double> Cf(double x) override;  

  double mu;
  double sigma;

  NormalDistribution(double mu, double sigma) : mu(mu), sigma(sigma) {};
  NormalDistribution() {};

};

RandomVariable<NormalDistribution> operator+(RandomVariable<NormalDistribution> const &var_a, RandomVariable<NormalDistribution> const &var_b);
RandomVariable<NormalDistribution> operator-(RandomVariable<NormalDistribution> const &var_a, RandomVariable<NormalDistribution> const &var_b);

RandomVariable<NormalDistribution> operator+(double var_a, RandomVariable<NormalDistribution> const &var_b);
RandomVariable<NormalDistribution> operator+(RandomVariable<NormalDistribution> const &var_a, double var_b);
RandomVariable<NormalDistribution> operator-(double var_a, RandomVariable<NormalDistribution> const &var_b);
RandomVariable<NormalDistribution> operator-(RandomVariable<NormalDistribution> const &var_a, double var_b);
RandomVariable<NormalDistribution> operator*(double var_a, RandomVariable<NormalDistribution> const &var_b);
RandomVariable<NormalDistribution> operator*(RandomVariable<NormalDistribution> const &var_a, double var_b);
RandomVariable<NormalDistribution> operator/(double var_a, RandomVariable<NormalDistribution> const &var_b);
RandomVariable<NormalDistribution> operator/(RandomVariable<NormalDistribution> const &var_a, double var_b);

RandomVariable<GenericDistribution> operator*(RandomVariable<NormalDistribution> const &var_a, RandomVariable<NormalDistribution> const &var_b);


}  // namespace random_variable
}  // namespace shrew