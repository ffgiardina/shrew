#pragma once

#include "random_variable.h"
#include <complex>

namespace shrew {
namespace random_variable {

class NormalDistribution : ProbabilityDistribution {
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

}  // namespace random_variable
}  // namespace shrew