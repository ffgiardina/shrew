#pragma once

#include "compound_distribution.h"
#include "../src/logic_assertions.h"
#include "random_variable.h"

namespace shrew
{
  namespace random_variable
  {
    /// @brief Normal (or Gaussian) probability distribution
    class NormalDistribution : public ProbabilityDistribution
    {
    public:
      // Probability density function
      virtual double Pdf(double x) const override;

      // Cumulative distribution function
      virtual double Cdf(double x) const override;

      // Moment generating function
      virtual double Mgf(double x) const override;

      // Characteristic function
      virtual std::complex<double> Cf(double x) const override;

      virtual std::tuple<const ProbabilityDistribution*, const ProbabilityDistribution*> get_operands() const override;

      double mu;
      double sigma;

      NormalDistribution const *l_operand;
      NormalDistribution const *r_operand;

      NormalDistribution(double mu, double sigma) : mu(mu), sigma(sigma), l_operand(0), r_operand(0) {};
      NormalDistribution(double mu, double sigma, NormalDistribution const *lptr, NormalDistribution  const *rptr) : mu(mu), sigma(sigma), l_operand(lptr), r_operand(rptr) 
      {
        std::unordered_set<const ProbabilityDistribution*> vars;
        if (has_repeating_random_variable(this, vars))
          throw std::logic_error("Error: Repeating random variable in compound expression detected. Arithmetic with correlated random variables not implemented. Try using constant expressions instead, e.g. X+X -> 2*X.");
      };
      NormalDistribution(){};
    };

    RandomVariable<NormalDistribution> operator+(double var_a, RandomVariable<NormalDistribution> const &var_b);
    RandomVariable<NormalDistribution> operator+(RandomVariable<NormalDistribution> const &var_a, double var_b);
    RandomVariable<NormalDistribution> operator-(double var_a, RandomVariable<NormalDistribution> const &var_b);
    RandomVariable<NormalDistribution> operator-(RandomVariable<NormalDistribution> const &var_a, double var_b);
    RandomVariable<NormalDistribution> operator*(double var_a, RandomVariable<NormalDistribution>const  &var_b);
    RandomVariable<NormalDistribution> operator*(RandomVariable<NormalDistribution> const &var_a, double var_b);
    RandomVariable<NormalDistribution> operator/(RandomVariable<NormalDistribution> const &var_a, double var_b);

    RandomVariable<NormalDistribution> operator+(RandomVariable<NormalDistribution> const &var_a, RandomVariable<NormalDistribution> const &var_b);
    RandomVariable<NormalDistribution> operator-(RandomVariable<NormalDistribution> const &var_a, RandomVariable<NormalDistribution> const &var_b);
  } // namespace random_variable
} // namespace shrew