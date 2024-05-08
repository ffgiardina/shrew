#define _USE_MATH_DEFINES
#include <math.h>
#include "generic_distribution.h"
#include "arithmetic.h"
#include <cmath>
#include <complex>

namespace shrew
{
    namespace random_variable
    {
        const numerical_methods::Integrator& GenericDistribution::generic_integrator = numerical_methods::GaussKronrod();

        double GenericDistribution::Pdf(double x)
        {
            return this->pdf(x);
        };

        double GenericDistribution::Cdf(double x)
        {
            return numerical_methods::cdf::compute(this->pdf, x, generic_integrator);
        };

        double GenericDistribution::Mgf(double t)
        {
            throw std::logic_error("Moment generating function not implemented for GenericDistribution");
        };

        std::complex<double> GenericDistribution::Cf(double t)
        {
            throw std::logic_error("Characteristic function not implemented for GenericDistribution");
        };

    } // namespace random_variable
} // namespace shrew