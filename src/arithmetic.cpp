#include <math.h>
#include "generic_distribution.h"
#include <functional>
#include <cmath>
#include "arithmetic.h"
#include "numerical_methods.h"

namespace shrew {
namespace random_variable {
namespace arithmetic {

namespace pdf {
numerical_methods::InfiniteDomainGaussKronrod integrator = numerical_methods::InfiniteDomainGaussKronrod();
double eval_random_variable_operation(double x, Operation operation, std::function<double(double)> l_eval, std::function<double(double)> r_eval, numerical_methods::Integrator &integrator )
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
        return abs(y) * l_eval(x * y) * r_eval(y);
    };

    auto exponentiation_integrand = [x, l_eval, r_eval](double y)
    {
        auto log_leval = [l_eval](double y) {return exp(y) * l_eval(exp(y));};
        return 1/abs(y) * log_leval(y) * r_eval(x / y);
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
            throw std::logic_error("Exponentiation of random variables not implemented");
    };

    throw std::logic_error("Operation not implemented");
};

double left_const_operation(double x, Operation operation, double l_eval, std::function<double(double)> r_eval )
{
    switch (operation) 
    {
        case addition:
            return l_eval + r_eval(x);
        case subtraction:
            return l_eval - r_eval(x);
        case multiplication:
            return 1/abs(l_eval) * r_eval(x / l_eval);
        case division:
            return 1/(abs(l_eval) * pow(x, 2)) * r_eval( 1 / (x * l_eval));
        case exponentiation:
            if (l_eval < 0)
            {
                throw std::logic_error("Negative exponentiation not implemented");
            }
            else
            {
                return abs(1 / (x * log(abs(l_eval)))) * r_eval(log(abs(x)) / log(abs(l_eval))) ;
            };
    };

    return 0.0;
};

double right_const_operation(double x, Operation operation, std::function<double(double)> l_eval, double r_eval )
{
    switch (operation) 
    {
        case addition:
            return l_eval(x) + r_eval;
        case subtraction:
            return l_eval(x) - r_eval;
        case multiplication:
            return 1/abs(r_eval) * l_eval(x / r_eval);
        case division:
            return abs(r_eval) * l_eval(x * r_eval);
        case exponentiation:
            throw std::logic_error("Negative exponentiation not implemented");
    };

    return 0.0;
}

}  // namespace pdf

namespace cdf {
numerical_methods::SemiInfiniteGaussKronrod integrator = numerical_methods::SemiInfiniteGaussKronrod(0.0);

double compute_cdf(std::function<double(double)> pdf, double x)
{
    integrator.upper_bound = x;
    return integrator.Integrate(pdf);
};

}  // namespace cdf
}  // namespace arithmetic
}  // namespace random_variable
}  // namespace shrew