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

double evaluate_pdf(double x, Operation operation, std::function<double(double)> l_eval, std::function<double(double)> r_eval)
{
    auto multiplication_integrand = [x, l_eval, r_eval](double y)
    {
        return 1/abs(y) * l_eval(y) * r_eval(x / y);
    };

    switch (operation) 
    {
        case addition:
            return 1.0;
        case subtraction:
            return 1.0;
        case multiplication:
            return integrator.Integrate(multiplication_integrand);
        case division:
            return 1.0;
        case exponentiation:
            return 1.0;
    }
};

}  // namespace arithmetic
}  // namespace random_variable
}  // namespace shrew