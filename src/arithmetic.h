#pragma once

#include "random_variable.h"
#include "numerical_methods.h"
#include <functional>
#include <complex>
#include <memory>

namespace shrew
{
    namespace random_variable
    {
        namespace arithmetic
        {

            enum Operation
            {
                addition,
                subtraction,
                multiplication,
                division,
                exponentiation
            };

            namespace evaluate_pdf
            {
                double random_variable_operation(double value, Operation operation, std::function<double(double)> l_eval, std::function<double(double)> r_eval, const numerical_methods::Integrator &integrator);
                double left_const_operation(double x, Operation operation, double l_eval, std::function<double(double)> r_eval);
                double right_const_operation(double x, Operation operation, std::function<double(double)> l_eval, double r_eval);
            } // namespace pdf

        } // namespace arithmetic
    }     // namespace random_variable
} // namespace shrew