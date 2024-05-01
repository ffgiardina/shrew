#pragma once

#include "random_variable.h"
#include <functional>
#include <complex>
#include <memory>

namespace shrew {
namespace random_variable {
namespace arithmetic {

enum Operation { addition, subtraction, multiplication, division, exponentiation };

double evaluate_pdf(double value, Operation operation, std::function<double(double)> l_eval, std::function<double(double)> r_eval);
  
}  // namespace arithmetic
}  // namespace random_variable
}  // namespace shrew