#pragma once

#include <complex>
#include <functional>

namespace shrew
{
  namespace numerical_methods
  {
    struct IntegratorConfig
    {
      static const unsigned int gauss_kronrod_n{211};
      static const unsigned int gauss_kronrod_max_depth{2};
      static const unsigned int mapped_gauss_kronrod_max_depth{0};
    };

    /// @brief Abstract interface for a numerical integrator
    class Integrator
    {
    public:
      // Computes the integration result over the mapped domain
      virtual double Integrate(std::function<double(double)> feval,
                               double lower_bound = -INFINITY,
                               double upper_bound = INFINITY) const = 0;
    };

    class GaussKronrod : public Integrator
    {
    public:
      virtual double Integrate(std::function<double(double)> feval,
                               double lower_limit,
                               double upper_limit) const override;
    };

    class MappedGaussKronrod : public Integrator
    {
    public:
      virtual double Integrate(std::function<double(double)> feval,
                               double lower_limit = 0,
                               double upper_limit = 1) const override;
      static const std::function<double(double)> MapDomain(
          std::function<double(double)> map);
    };

    namespace cdf
    {
      double compute(std::function<double(double)> pdf, double x,
                     numerical_methods::Integrator const &integrator);
    } // namespace cdf
  } // namespace numerical_methods
} // namespace shrew