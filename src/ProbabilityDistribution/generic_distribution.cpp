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

        double GenericDistribution::Pdf(double x)
        {
            return this->pdf(x);
        };

        double GenericDistribution::Cdf(double x)
        {
            return arithmetic::cdf::compute(this->pdf, x);
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