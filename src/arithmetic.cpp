#include "shrew/arithmetic.hpp"

#include <math.h>

namespace shrew
{
  namespace random_variable
  {
    namespace arithmetic
    {

      namespace evaluate_pdf
      {
        double random_variable_operation(
            double x, Operation operation, std::function<double(double)> l_eval,
            std::function<double(double)> r_eval,
            const numerical_methods::Integrator &integrator)
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
            return 1.0 / abs(y) * l_eval(y) * r_eval(x / y);
          };

          auto division_integrand = [x, l_eval, r_eval](double y)
          {
            return abs(y) * l_eval(x * y) * r_eval(y);
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
            throw std::logic_error(
                "Exponentiation of random variables not implemented");
          };

          throw std::logic_error("Operation not implemented");
        };

        double left_const_operation(double x, Operation operation, double l_eval,
                                    std::function<double(double)> r_eval)
        {
          switch (operation)
          {
          case addition:
            return r_eval(x - l_eval);
          case subtraction:
            return r_eval(x + l_eval);
          case multiplication:
            return 1 / abs(l_eval) * r_eval(x / l_eval);
          case division:
            return 1 / (abs(l_eval) * pow(x, 2)) * r_eval(1 / (x * l_eval));
          case exponentiation:
            if (l_eval < 0)
            {
              throw std::logic_error(
                  "Negative base for random variable exponentiation");
            }
            else if (x < 0)
            {
              throw std::logic_error(
                  "Pdf of base with random variable exponent not defined for "
                  "negative values");
            }
            else
            {
              return abs(1 / (x * log(l_eval))) * r_eval(log(x) / log(l_eval));
            }
          };

          return 0.0;
        };

        double right_const_operation(double x, Operation operation,
                                     std::function<double(double)> l_eval,
                                     double r_eval)
        {
          switch (operation)
          {
          case addition:
            return l_eval(x - r_eval);
          case subtraction:
            return l_eval(x + r_eval);
          case multiplication:
            return 1 / abs(r_eval) * l_eval(x / r_eval);
          case division:
            return abs(r_eval) * l_eval(x * r_eval);
          case exponentiation:
            if (int(r_eval) != r_eval)
            {
              throw std::logic_error("Exponent must be an integer");
            }
            else if (int(r_eval) % 2 == 0 && x < 0)
            {
              throw std::logic_error(
                  "Pdf of random variable constructed with even exponent not defined "
                  "for negative values");
            }
            else
            {
              int multiplier = (1.0 + int(abs(r_eval) + 1) % 2);
              int sign = (x >= 0) ? 1 : -1;
              return l_eval(sign * pow(abs(x), 1.0 / r_eval)) *
                     abs(multiplier / r_eval * sign *
                         pow(abs(x), 1.0 / r_eval - 1.0));
            }
          };

          return 0.0;
        }

      } // namespace evaluate_pdf

    } // namespace arithmetic
  } // namespace random_variable
} // namespace shrew