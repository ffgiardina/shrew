#pragma once

#include "../src/logic_assertions.h"
#include "compound_distribution.h"
#include "random_variable.h"

namespace shrew {
namespace random_variable {

template <typename T>
using RV = RandomVariable<T>;

/// @brief Normal (or Gaussian) probability distribution
class NormalDistribution : public ProbabilityDistribution {
 public:
  /// Point-wise probability density function
  virtual double Pdf(double x) const override;

  /// Point-wise cumulative distribution function
  virtual double Cdf(double x) const override;

  /// Point-wise moment generating function
  virtual double Mgf(double x) const override;

  /// Point-wise characteristic function
  virtual std::complex<double> Cf(double x) const override;

  NormalDistribution(double mu, double sigma)
      : mu(mu), sigma(sigma), l_operand(0), r_operand(0){};
  NormalDistribution(){};

 private:
  double mu;
  double sigma;

  NormalDistribution const *l_operand;
  NormalDistribution const *r_operand;

  /// Gets left and right operands if they are of type ProbabilityDistribution,
  /// otherwise returns null pointer.
  virtual std::tuple<const ProbabilityDistribution *,
                     const ProbabilityDistribution *>
  get_pd_operands() const override;

  /// Constructs a compound distribution. Throws an error if repeating random
  /// variables are detected in the binary operation tree, as correlation is not
  /// yet taken into account in the integrator.
  NormalDistribution(double mu, double sigma, NormalDistribution const *lptr,
                     NormalDistribution const *rptr)
      : mu(mu), sigma(sigma), l_operand(lptr), r_operand(rptr) {
    std::unordered_set<const ProbabilityDistribution *> vars;
    if (LogicAssertions::has_repeating_random_variable(this, vars))
      throw std::logic_error(
          "Error: Repeating random variable in compound expression detected. "
          "Arithmetic with correlated random variables not implemented. Try "
          "using constant expressions instead, e.g. X+X -> 2*X.");
  };

  friend class LogicAssertions;
  
  friend RV<NormalDistribution> operator+(RV<NormalDistribution> const &var_a,
                                          RV<NormalDistribution> const &var_b);
  friend RV<NormalDistribution> operator-(RV<NormalDistribution> const &var_a,
                                          RV<NormalDistribution> const &var_b);
  friend RV<NormalDistribution> operator+(double var_a,
                                          RV<NormalDistribution> const &var_b);
  friend RV<NormalDistribution> operator+(RV<NormalDistribution> const &var_a,
                                          double var_b);
  friend RV<NormalDistribution> operator-(double var_a,
                                          RV<NormalDistribution> const &var_b);
  friend RV<NormalDistribution> operator-(RV<NormalDistribution> const &var_a,
                                          double var_b);
  friend RV<NormalDistribution> operator*(double var_a,
                                          RV<NormalDistribution> const &var_b);
  friend RV<NormalDistribution> operator*(RV<NormalDistribution> const &var_a,
                                          double var_b);
  friend RV<NormalDistribution> operator/(RV<NormalDistribution> const &var_a,
                                          double var_b);
};

RandomVariable<NormalDistribution> operator+(
    double var_a, RandomVariable<NormalDistribution> const &var_b);
RandomVariable<NormalDistribution> operator+(
    RandomVariable<NormalDistribution> const &var_a, double var_b);
RandomVariable<NormalDistribution> operator-(
    double var_a, RandomVariable<NormalDistribution> const &var_b);
RandomVariable<NormalDistribution> operator-(
    RandomVariable<NormalDistribution> const &var_a, double var_b);
RandomVariable<NormalDistribution> operator*(
    double var_a, RandomVariable<NormalDistribution> const &var_b);
RandomVariable<NormalDistribution> operator*(
    RandomVariable<NormalDistribution> const &var_a, double var_b);
RandomVariable<NormalDistribution> operator/(
    RandomVariable<NormalDistribution> const &var_a, double var_b);

RandomVariable<NormalDistribution> operator+(
    RandomVariable<NormalDistribution> const &var_a,
    RandomVariable<NormalDistribution> const &var_b);
RandomVariable<NormalDistribution> operator-(
    RandomVariable<NormalDistribution> const &var_a,
    RandomVariable<NormalDistribution> const &var_b);
}  // namespace random_variable
}  // namespace shrew