#pragma once

#include "random_variable.h"
#include "numerical_methods.h"
#include <functional>
#include <complex>
#include <memory>

namespace shrew {
namespace random_variable {
namespace arithmetic {

enum Operation { addition, subtraction, multiplication, division, exponentiation };
inline numerical_methods::InfiniteDomainGaussKronrod default_integrator = numerical_methods::InfiniteDomainGaussKronrod();

namespace pdf {
double eval_random_variable_operation(double value, Operation operation, std::function<double(double)> l_eval, std::function<double(double)> r_eval, numerical_methods::Integrator &integrator = default_integrator);
double left_const_operation(double x, Operation operation, double l_eval, std::function<double(double)> r_eval );
double right_const_operation(double x, Operation operation, std::function<double(double)> l_eval, double r_eval );
}  // namespace pdf

namespace cdf {
double compute_cdf(std::function<double(double)> pdf, double x);
}  // namespace cdf

}  // namespace arithmetic
}  // namespace random_variable
}  // namespace shrew