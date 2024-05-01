#pragma once

#include "random_variable.h"
#include "arithmetic.h"
#include <complex>
#include <memory>

namespace shrew {
namespace random_variable {

/// @brief Generic probability distribution 
class GenericDistribution : public ProbabilityDistribution {
 public:
  // Probability density function
  virtual double Pdf(double x) override;

  // Cumulative distribution function
  virtual double Cdf(double x) override;

  // Moment generating function
  virtual double Mgf(double x) override;

  // Characteristic function
  virtual std::complex<double> Cf(double x) override;  

  std::shared_ptr<ProbabilityDistribution> l_operand;
  std::shared_ptr<ProbabilityDistribution> r_operand;
  arithmetic::Operation operation;

  GenericDistribution(std::shared_ptr<ProbabilityDistribution> lptr, std::shared_ptr<ProbabilityDistribution> rptr, arithmetic::Operation operation) : l_operand(lptr), r_operand(rptr), operation(operation) {};
  GenericDistribution() {};

};

RandomVariable<GenericDistribution> operator+(RandomVariable<GenericDistribution> const &var_a, RandomVariable<GenericDistribution> const &var_b);
RandomVariable<GenericDistribution> operator-(RandomVariable<GenericDistribution> const &var_a, RandomVariable<GenericDistribution> const &var_b);

}  // namespace random_variable
}  // namespace shrew