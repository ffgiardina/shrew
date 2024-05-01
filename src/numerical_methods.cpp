#include <math.h>
#include <functional>
#include <cmath>
#include "numerical_methods.h"
#include <boost/math/quadrature/gauss_kronrod.hpp>

namespace shrew {
namespace numerical_methods {

std::function<double(double)> InfiniteDomainGaussKronrod::MapDomain(std::function<double(double)> map)
{
    return [map](double x) { return (map(1/x - 1) + map(-1/x + 1)) / pow(x, 2);};
};

double InfiniteDomainGaussKronrod::Integrate(std::function<double(double)> feval)
{
    double  error;
    return boost::math::quadrature::gauss_kronrod<double, n_point>::integrate(this->MapDomain(feval), 0.0, 1.0, 0, 0, &error);
};

}  // namespace numerical_methods
}  // namespace shrew