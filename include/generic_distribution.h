#pragma once

#include <functional>

#include "../src/numerical_methods.h"
#include "random_variable.h"

namespace shrew
{
  namespace random_variable
  {

    /// @brief Generic probability distribution
    /// User can provide a custom probability density function. No checks are being
    /// performed and it is assumed the pdf is a valid probability density function.
    class GenericDistribution : public ProbabilityDistribution
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

      /// Constructs a generic distribution with a user-provided probability density
      /// function
      GenericDistribution(std::function<double(double)> pdf) : pdf(pdf) {};
      GenericDistribution() {};

    private:
      /// User-provided probability density function
      std::function<double(double)> pdf;

      /// Integrator used for the cumulative distribution function
      static const numerical_methods::Integrator &generic_integrator;
    };

  } // namespace random_variable
} // namespace shrew