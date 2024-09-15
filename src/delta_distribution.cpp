#include "shrew/delta_distribution.hpp"

#include <complex>

#include "shrew/arithmetic.hpp"

namespace shrew
{
  namespace random_variable
  {

    double DeltaDistribution::Pdf(double x) const
    {
      throw std::logic_error(
          "Probability density function not implemented for DeltaDistribution");
    };
    double DeltaDistribution::Cdf(double x) const { return x > value ? 1.0 : 0.0; };

    double DeltaDistribution::Mgf(double t) const
    {
      throw std::logic_error(
          "Moment generating function not implemented for GenericDistribution");
    };

    std::complex<double> DeltaDistribution::Cf(double t) const
    {
      throw std::logic_error(
          "Characteristic function not implemented for GenericDistribution");
    };

  } // namespace random_variable
} // namespace shrew