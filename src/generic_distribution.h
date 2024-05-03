#pragma once

#include "random_variable.h"
#include "arithmetic.h"
#include <complex>
#include <memory>
#include <cmath>
#include <math.h>

namespace shrew {
namespace random_variable {

/// @brief Generic probability distribution 
/// @tparam T 
/// @tparam U 
template<typename T, typename U>
class GenericDistribution : public ProbabilityDistribution {
 public:
  // Probability density function
  virtual double Pdf(double x) override;

  // Cumulative distribution function
  virtual double Cdf(double x) override;

  // Moment generating function
  virtual double Mgf(double x) override;

  // Characteristic function
  virtual std::complex<double> Cf(double x) override;  

  std::shared_ptr<T> l_operand;
  std::shared_ptr<U> r_operand;
  arithmetic::Operation operation;

  GenericDistribution(std::shared_ptr<T> lptr, std::shared_ptr<U> rptr, arithmetic::Operation operation) : l_operand(lptr), r_operand(rptr), operation(operation) {};
  GenericDistribution() {};
};

template<typename T>
class GenericDistribution<T, double> : public ProbabilityDistribution {
 public:
  // Probability density function
  virtual double Pdf(double x) override;

  // Cumulative distribution function
  virtual double Cdf(double x) override;

  // Moment generating function
  virtual double Mgf(double x) override;

  // Characteristic function
  virtual std::complex<double> Cf(double x) override;  

  std::shared_ptr<T> l_operand;
  double r_operand;
  arithmetic::Operation operation;

  GenericDistribution(std::shared_ptr<T> lptr, double rptr, arithmetic::Operation operation) : l_operand(lptr), r_operand(rptr), operation(operation) {};
  GenericDistribution() {};
};

template<typename U>
class GenericDistribution<double, U> : public ProbabilityDistribution {
 public:
  // Probability density function
  virtual double Pdf(double x) override;

  // Cumulative distribution function
  virtual double Cdf(double x) override;

  // Moment generating function
  virtual double Mgf(double x) override;

  // Characteristic function
  virtual std::complex<double> Cf(double x) override;  

  double l_operand;
  std::shared_ptr<U> r_operand;
  arithmetic::Operation operation;

  GenericDistribution(double lptr, std::shared_ptr<U> rptr, arithmetic::Operation operation) : l_operand(lptr), r_operand(rptr), operation(operation) {};
  GenericDistribution() {};
};

template<typename T, typename U>
double GenericDistribution<T, U>::Pdf(double x) 
{
    return arithmetic::pdf::eval_random_variable_operation(x, operation, [this](double t) {return l_operand->Pdf(t); }, [this](double t) {return r_operand->Pdf(t); });
};

template<typename T>
double GenericDistribution<T, double>::Pdf(double x) 
{
    return arithmetic::pdf::right_const_operation(x, operation, [this](double t) {return l_operand->Pdf(t); }, r_operand);
};

template<typename U>
double GenericDistribution<double, U>::Pdf(double x) 
{
    return arithmetic::pdf::left_const_operation(x, operation, l_operand, [this](double t) {return r_operand->Pdf(t); });
};

template<typename T, typename U>
double GenericDistribution<T, U>::Cdf(double x) 
{
    throw std::logic_error("Method not implemented");
};

template<typename T>
double GenericDistribution<T, double>::Cdf(double x) 
{
    throw std::logic_error("Method not implemented");
};

template<typename U>
double GenericDistribution<double, U>::Cdf(double x) 
{
    throw std::logic_error("Method not implemented");
};

template<typename T, typename U>
double GenericDistribution<T, U>::Mgf(double t) 
{
    throw std::logic_error("Method not implemented");
};

template<typename T>
double GenericDistribution<T, double>::Mgf(double x) 
{
    throw std::logic_error("Method not implemented");
};

template<typename U>
double GenericDistribution<double, U>::Mgf(double x) 
{
    throw std::logic_error("Method not implemented");
};

template<typename T, typename U>
std::complex<double> GenericDistribution<T, U>::Cf(double t) 
{
    throw std::logic_error("Method not implemented");
};

template<typename T>
std::complex<double> GenericDistribution<T, double>::Cf(double x) 
{
    throw std::logic_error("Method not implemented");
};

template<typename U>
std::complex<double> GenericDistribution<double, U>::Cf(double x) 
{
    throw std::logic_error("Method not implemented");
};

template<typename T, typename U>
RandomVariable<GenericDistribution<T, U>> operator+(RandomVariable<T> const &var_a, RandomVariable<U> const &var_b)
{
    return GenericDistribution(std::make_shared<T>(var_a.probability_distribution), std::make_shared<U>(var_b.probability_distribution), arithmetic::addition);
};

template<typename T, typename U>
RandomVariable<GenericDistribution<T, U>> operator-(RandomVariable<T> const &var_a, RandomVariable<U> const &var_b)
{
    return GenericDistribution(std::make_shared<T>(var_a.probability_distribution), std::make_shared<U>(var_b.probability_distribution), arithmetic::subtraction);
};

template<typename T, typename U>
RandomVariable<GenericDistribution<T, U>> operator*(RandomVariable<T> const &var_a, RandomVariable<U> const &var_b)
{
    return GenericDistribution(std::make_shared<T>(var_a.probability_distribution), std::make_shared<U>(var_b.probability_distribution), arithmetic::multiplication);
};

template<typename T, typename U>
RandomVariable<GenericDistribution<T, U>> operator/(RandomVariable<T> const &var_a, RandomVariable<U> const &var_b)
{
    return GenericDistribution(std::make_shared<T>(var_a.probability_distribution), std::make_shared<U>(var_b.probability_distribution), arithmetic::division);
};

template<typename T, typename U>
RandomVariable<GenericDistribution<T, U>> operator^(RandomVariable<T> const &var_a, RandomVariable<U> const &var_b)
{
    return GenericDistribution(std::make_shared<T>(var_a.probability_distribution), std::make_shared<U>(var_b.probability_distribution), arithmetic::exponentiation);
};
}  // namespace random_variable
}  // namespace shrew