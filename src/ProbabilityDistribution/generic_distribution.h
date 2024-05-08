#pragma once

#include "random_variable.h"
#include "compound_distribution.h"
#include <complex>

namespace shrew
{
  namespace random_variable
  {

    /// @brief Generic probability distribution
    class GenericDistribution : public ProbabilityDistribution
    {
    public:
      // Probability density function
      virtual double Pdf(double x) override;

      // Cumulative distribution function
      virtual double Cdf(double x) override;

      // Moment generating function
      virtual double Mgf(double x) override;

      // Characteristic function
      virtual std::complex<double> Cf(double x) override;

      std::function<double(double)> pdf;

      GenericDistribution(std::function<double(double)> pdf) : pdf(pdf){};
      GenericDistribution(){};
    };

  } // namespace random_variable
} // namespace shrew