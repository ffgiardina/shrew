#include <math.h>
#include "generic_distribution.h"
#include <functional>
#include <cmath>
#include "arithmetic.h"
#include "numerical_methods.h"

namespace shrew {
namespace random_variable {
namespace arithmetic {

// TODO: think about injection of the integrator
numerical_methods::InfiniteDomainGaussKronrod integrator = numerical_methods::InfiniteDomainGaussKronrod();

double evaluate_pdf(double x, Operation operation, std::function<double(double)> l_eval, std::function<double(double)> r_eval, numerical_methods::Integrator &integrator )
{
    auto addition_integrand = [x, l_eval, r_eval](double y)
    {
        return l_eval(y) * r_eval(x - y);
    };

    auto subtraction_integrand = [x, l_eval, r_eval](double y)
    {
        return l_eval(y) * r_eval(y - x);
    };

    auto multiplication_integrand = [x, l_eval, r_eval](double y)
    {
        return 1/abs(y) * l_eval(y) * r_eval(x / y);
    };

    auto division_integrand = [x, l_eval, r_eval](double y)
    {
        return abs(y) * l_eval(y) * r_eval(x * y);
    };

    switch (operation) 
    {
        case addition:
            return integrator.Integrate(addition_integrand);
        case subtraction:
            return integrator.Integrate(subtraction_integrand);
        case multiplication:
            return integrator.Integrate(multiplication_integrand);
        case division:
            return integrator.Integrate(division_integrand);
        case exponentiation:
            return 1.0;
    }
};

}  // namespace arithmetic
}  // namespace random_variable
}  // namespace shrew