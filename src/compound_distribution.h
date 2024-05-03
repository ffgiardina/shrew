#pragma once

#include "random_variable.h"
#include "arithmetic.h"
#include <complex>
#include <memory>
#include <cmath>
#include <math.h>

namespace shrew {
namespace random_variable {

/// @brief Compound probability distribution created from two underlying distributions
/// @tparam T 
/// @tparam U 
template<typename T, typename U>
class CompoundDistribution : public ProbabilityDistribution {
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

  CompoundDistribution(std::shared_ptr<T> lptr, std::shared_ptr<U> rptr, arithmetic::Operation operation) : l_operand(lptr), r_operand(rptr), operation(operation) {};
  CompoundDistribution() {};
};

template<typename T>
class CompoundDistribution<T, double> : public ProbabilityDistribution {
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

  CompoundDistribution(std::shared_ptr<T> lptr, double rptr, arithmetic::Operation operation) : l_operand(lptr), r_operand(rptr), operation(operation) {};
  CompoundDistribution() {};
};

template<typename U>
class CompoundDistribution<double, U> : public ProbabilityDistribution {
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

  CompoundDistribution(double lptr, std::shared_ptr<U> rptr, arithmetic::Operation operation) : l_operand(lptr), r_operand(rptr), operation(operation) {};
  CompoundDistribution() {};
};

template<typename T, typename U>
double CompoundDistribution<T, U>::Pdf(double x) 
{
    return arithmetic::pdf::eval_random_variable_operation(x, operation, [this](double t) {return l_operand->Pdf(t); }, [this](double t) {return r_operand->Pdf(t); });
};

template<typename T>
double CompoundDistribution<T, double>::Pdf(double x) 
{
    return arithmetic::pdf::right_const_operation(x, operation, [this](double t) {return l_operand->Pdf(t); }, r_operand);
};

template<typename U>
double CompoundDistribution<double, U>::Pdf(double x) 
{
    return arithmetic::pdf::left_const_operation(x, operation, l_operand, [this](double t) {return r_operand->Pdf(t); });
};

template<typename T, typename U>
double CompoundDistribution<T, U>::Cdf(double x) 
{
    return arithmetic::cdf::compute([this](double y) {return this->Pdf(y);}, x);
};

template<typename T>
double CompoundDistribution<T, double>::Cdf(double x) 
{
    return arithmetic::cdf::compute([this](double y) {return this->Pdf(y);}, x);
};

template<typename U>
double CompoundDistribution<double, U>::Cdf(double x) 
{
    return arithmetic::cdf::compute([this](double y) {return this->Pdf(y);}, x);
};

template<typename T, typename U>
double CompoundDistribution<T, U>::Mgf(double t) 
{
    throw std::logic_error("Moment generating function not implemented for CompoundDistribution");
};

template<typename T>
double CompoundDistribution<T, double>::Mgf(double x) 
{
    throw std::logic_error("Moment generating function not implemented for CompoundDistribution");
};

template<typename U>
double CompoundDistribution<double, U>::Mgf(double x) 
{
    throw std::logic_error("Moment generating function not implemented for CompoundDistribution");
};

template<typename T, typename U>
std::complex<double> CompoundDistribution<T, U>::Cf(double t) 
{
    throw std::logic_error("Characteristic function not implemented for CompoundDistribution");
};

template<typename T>
std::complex<double> CompoundDistribution<T, double>::Cf(double x) 
{
    throw std::logic_error("Characteristic function not implemented for CompoundDistribution");

};

template<typename U>
std::complex<double> CompoundDistribution<double, U>::Cf(double x) 
{
    throw std::logic_error("Characteristic function not implemented for CompoundDistribution");
};

template<typename T, typename U>
RandomVariable<CompoundDistribution<T, U>> operator+(RandomVariable<T> const &var_a, RandomVariable<U> const &var_b)
{
    return CompoundDistribution(std::make_shared<T>(var_a.probability_distribution), std::make_shared<U>(var_b.probability_distribution), arithmetic::addition);
};

template<typename T, typename U>
RandomVariable<CompoundDistribution<T, U>> operator-(RandomVariable<T> const &var_a, RandomVariable<U> const &var_b)
{
    return CompoundDistribution(std::make_shared<T>(var_a.probability_distribution), std::make_shared<U>(var_b.probability_distribution), arithmetic::subtraction);
};

template<typename T, typename U>
RandomVariable<CompoundDistribution<T, U>> operator*(RandomVariable<T> const &var_a, RandomVariable<U> const &var_b)
{
    return CompoundDistribution(std::make_shared<T>(var_a.probability_distribution), std::make_shared<U>(var_b.probability_distribution), arithmetic::multiplication);
};

template<typename T, typename U>
RandomVariable<CompoundDistribution<T, U>> operator/(RandomVariable<T> const &var_a, RandomVariable<U> const &var_b)
{
    return CompoundDistribution(std::make_shared<T>(var_a.probability_distribution), std::make_shared<U>(var_b.probability_distribution), arithmetic::division);
};

template<typename T, typename U>
RandomVariable<CompoundDistribution<T, U>> operator^(RandomVariable<T> const &var_a, RandomVariable<U> const &var_b)
{
    return CompoundDistribution(std::make_shared<T>(var_a.probability_distribution), std::make_shared<U>(var_b.probability_distribution), arithmetic::exponentiation);
};
}  // namespace random_variable
}  // namespace shrew