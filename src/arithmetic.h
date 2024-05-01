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
double evaluate_pdf(double value, Operation operation, std::function<double(double)> l_eval, std::function<double(double)> r_eval, numerical_methods::Integrator &integrator = default_integrator);
  
}  // namespace arithmetic
}  // namespace random_variable
}  // namespace shrew