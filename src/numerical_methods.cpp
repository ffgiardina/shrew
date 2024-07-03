#include "numerical_methods.h"

#include <boost/math/quadrature/gauss_kronrod.hpp>

namespace shrew
{
  namespace numerical_methods
  {
    double GaussKronrod::Integrate(std::function<double(double)> feval,
                                   double lower_limit, double upper_limit) const
    {
      return boost::math::quadrature::
          gauss_kronrod<double, IntegratorConfig::gauss_kronrod_n>::integrate(
              feval, lower_limit, upper_limit,
              IntegratorConfig::gauss_kronrod_max_depth);
    };

    const std::function<double(double)> MappedGaussKronrod::MapDomain(
        std::function<double(double)> map)
    {
      return [map](double x)
      {
        return (map(1 / x - 1) + map(-1 / x + 1)) / pow(x, 2);
      };
    };

    double MappedGaussKronrod::Integrate(std::function<double(double)> feval,
                                         double lower_limit,
                                         double upper_limit) const
    {
      return boost::math::quadrature::
          gauss_kronrod<double, IntegratorConfig::gauss_kronrod_n>::integrate(
              MappedGaussKronrod::MapDomain(feval), 0, 1,
              IntegratorConfig::mapped_gauss_kronrod_max_depth);
    };

    namespace cdf
    {
      double compute(std::function<double(double)> pdf, double x,
                     numerical_methods::Integrator const &integrator)
      {
        return integrator.Integrate(pdf, -INFINITY, x);
      };

    } // namespace cdf
  } // namespace numerical_methods
} // namespace shrew