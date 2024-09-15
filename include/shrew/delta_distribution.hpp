#pragma once

#include <functional>

#include "shrew/random_variable.hpp"

namespace shrew
{
  namespace random_variable
  {

    /// @brief Dirac delta probability distribution
    /// This class allows for constants that derive from ProbabilityDistribution if
    /// needed.
    class DeltaDistribution : public ProbabilityDistribution
    {
    public:
      /// Point-wise probability density function
      virtual double Pdf(double x) const override;

      /// Point-wise cumulative distribution function
      virtual double Cdf(double x) const override;

      /// Point-wise moment generating function
      virtual double Mgf(double x) const override;

      /// Point-wise characteristic function
      virtual std::complex<double> Cf(double x) const override;

      double get_value() const { return value; };
      DeltaDistribution(double value) : value(value) {};
      DeltaDistribution() {};

    private:
      /// Dirac delta peak location
      double value;
    };

  } // namespace random_variable
} // namespace shrew