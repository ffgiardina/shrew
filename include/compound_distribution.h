#pragma once

#include <tuple>
#include <unordered_set>

#include "../src/arithmetic.h"
#include "../src/logic_assertions.h"
#include "delta_distribution.h"
#include "random_variable.h"

namespace shrew {
namespace random_variable {

template <typename T>
using RV = RandomVariable<T>;
using PD = ProbabilityDistribution;

/// @brief Compound probability distribution created from a binary operation
/// with two random variables.
/// @tparam T
///  The type of the left operand.
/// @tparam U
///  The type of the right operand.
template <typename T, typename U>
class CompoundDistribution : public PD {
 public:
  /// Point-wise probability density function
  virtual double Pdf(double x) const override;

  /// Point-wise cumulative distribution function
  virtual double Cdf(double x) const override;

  /// Point-wise moment generating function
  virtual double Mgf(double x) const override;

  /// Point-wise characteristic function
  virtual std::complex<double> Cf(double x) const override;

  CompoundDistribution(){};

 private:
  /// The left operand of which the distribution is composed.
  T const *l_operand;

  /// The right operand of which the distribution is composed.
  U const *r_operand;

  /// The operation between the left and right operand
  arithmetic::Operation operation;

  /// Integrator used for the compound PDF expression
  static const numerical_methods::Integrator &compound_integrator;

  /// Integrator used for the compound CDF expression
  static const numerical_methods::Integrator &cdf_integrator;

  /// Gets left and right operands if they are of type PD,
  /// otherwise returns null pointer.
  virtual std::tuple<const PD *, const PD *> get_pd_operands() const override;

  /// Constructs a compound distribution. Throws an error if repeating random
  /// variables are detected in the binary operation tree, as correlation is not
  /// yet taken into account in the integrator.
  CompoundDistribution(T const *lptr, U const *rptr,
                       arithmetic::Operation operation)
      : l_operand(lptr), r_operand(rptr), operation(operation) {
    std::unordered_set<const PD *> vars;

    if (LogicAssertions::has_repeating_random_variable(this, vars))
      throw std::logic_error(
          "Error: Repeating random variable in compound expression detected. "
          "Arithmetic with correlated random variables not implemented. Try "
          "using constant expressions instead, e.g. X+X -> 2*X.");
  };

  friend class LogicAssertions;

  template <typename L, typename R>
  friend RV<CompoundDistribution<L, R>> operator+(RV<L> const &var_a,
                                                  RV<R> const &var_b);
  template <typename L, typename R>
  friend RV<CompoundDistribution<L, R>> operator-(RV<L> const &var_a,
                                                  RV<R> const &var_b);
  template <typename L, typename R>
  friend RV<CompoundDistribution<L, R>> operator*(RV<L> const &var_a,
                                                  RV<R> const &var_b);
  template <typename L, typename R>
  friend RV<CompoundDistribution<L, R>> operator/(RV<L> const &var_a,
                                                  RV<R> const &var_b);
  template <typename L, typename R>
  friend RV<CompoundDistribution<L, R>> operator^(RV<L> const &var_a,
                                                  RV<R> const &var_b);
};

/// @brief Compound probability distribution created from two underlying
/// distributions. Specialized for right-constant arithmetic operations.
/// @tparam T
///  The type of the left operand
template <typename T>
class CompoundDistribution<T, double> : public PD {
 public:
  /// Point-wise probability density function
  virtual double Pdf(double x) const override;

  /// Point-wise cumulative distribution function
  virtual double Cdf(double x) const override;

  /// Point-wise moment generating function
  virtual double Mgf(double x) const override;

  /// Point-wise characteristic function
  virtual std::complex<double> Cf(double x) const override;

  CompoundDistribution(){};

 private:
  T const *l_operand;
  double r_operand;
  arithmetic::Operation operation;

  static const numerical_methods::Integrator &compound_integrator;
  static const numerical_methods::Integrator &cdf_integrator;

  virtual std::tuple<const PD *, const PD *> get_pd_operands() const override;

  CompoundDistribution(T const *lptr, double rptr,
                       arithmetic::Operation operation)
      : l_operand(lptr), r_operand(rptr), operation(operation){};

  friend class LogicAssertions;

  template <typename L>
  friend RV<CompoundDistribution<L, double>> operator+(RV<L> const &var_a,
                                                       double var_b);
  template <typename L>
  friend RV<CompoundDistribution<L, double>> operator-(RV<L> const &var_a,
                                                       double var_b);
  template <typename L>
  friend RV<CompoundDistribution<L, double>> operator*(RV<L> const &var_a,
                                                       double var_b);
  template <typename L>
  friend RV<CompoundDistribution<L, double>> operator/(RV<L> const &var_a,
                                                       double var_b);
  template <typename L>
  friend RV<CompoundDistribution<L, double>> operator^(RV<L> const &var_a,
                                                       double var_b);
};

/// @brief Compound probability distribution created from two underlying
/// distributions. Specialized for left-constant arithmetic operations.
/// @tparam U
///  The type of the right operand
template <typename U>
class CompoundDistribution<double, U> : public PD {
 public:
  /// Point-wise probability density function
  virtual double Pdf(double x) const override;

  /// Point-wise cumulative distribution function
  virtual double Cdf(double x) const override;

  /// Point-wise moment generating function
  virtual double Mgf(double x) const override;

  /// Point-wise characteristic function
  virtual std::complex<double> Cf(double x) const override;

  CompoundDistribution(){};

 private:
  double l_operand;
  U const *r_operand;
  arithmetic::Operation operation;

  static const numerical_methods::Integrator &compound_integrator;
  static const numerical_methods::Integrator &cdf_integrator;

  virtual std::tuple<const PD *, const PD *> get_pd_operands() const override;

  CompoundDistribution(double lptr, U const *rptr,
                       arithmetic::Operation operation)
      : l_operand(lptr), r_operand(rptr), operation(operation){};

  friend class LogicAssertions;
  template <typename R>
  friend RV<CompoundDistribution<double, R>> operator+(double var_a,
                                                       RV<R> const &var_b);
  template <typename R>
  friend RV<CompoundDistribution<double, R>> operator-(double var_a,
                                                       RV<R> const &var_b);
  template <typename R>
  friend RV<CompoundDistribution<double, R>> operator*(double var_a,
                                                       RV<R> const &var_b);
  template <typename R>
  friend RV<CompoundDistribution<double, R>> operator/(double var_a,
                                                       RV<R> const &var_b);
  template <typename R>
  friend RV<CompoundDistribution<double, R>> operator^(double var_a,
                                                       RV<R> const &var_b);
};

template <typename T, typename U>
std::tuple<const PD *, const PD *> CompoundDistribution<T, U>::get_pd_operands()
    const {
  return std::make_tuple(l_operand, r_operand);
};

template <typename U>
std::tuple<const PD *, const PD *>
CompoundDistribution<double, U>::get_pd_operands() const {
  std::tuple<const U *, const U *> tup(0, r_operand);
  return tup;
};

template <typename T>
std::tuple<const PD *, const PD *>
CompoundDistribution<T, double>::get_pd_operands() const {
  std::tuple<const T *, const T *> tup(l_operand, 0);
  return tup;
};

template <typename T, typename U>
const numerical_methods::Integrator
    &CompoundDistribution<T, U>::compound_integrator =
        numerical_methods::MappedGaussKronrod();

template <typename T>
const numerical_methods::Integrator
    &CompoundDistribution<T, double>::compound_integrator =
        numerical_methods::MappedGaussKronrod();

template <typename U>
const numerical_methods::Integrator
    &CompoundDistribution<double, U>::compound_integrator =
        numerical_methods::MappedGaussKronrod();

template <typename T, typename U>
const numerical_methods::Integrator
    &CompoundDistribution<T, U>::cdf_integrator =
        numerical_methods::GaussKronrod();

template <typename T>
const numerical_methods::Integrator
    &CompoundDistribution<T, double>::cdf_integrator =
        numerical_methods::GaussKronrod();

template <typename U>
const numerical_methods::Integrator
    &CompoundDistribution<double, U>::cdf_integrator =
        numerical_methods::GaussKronrod();

template <typename T, typename U>
double CompoundDistribution<T, U>::Pdf(double x) const {
    return arithmetic::evaluate_pdf::random_variable_operation(
        x, operation, [this](double t) { return l_operand->Pdf(t); },
        [this](double t) { return r_operand->Pdf(t); }, compound_integrator);
};

template <typename T>
double CompoundDistribution<T, double>::Pdf(double x) const {
  return arithmetic::evaluate_pdf::right_const_operation(
      x, operation, [this](double t) { return l_operand->Pdf(t); }, r_operand);
};

template <typename U>
double CompoundDistribution<double, U>::Pdf(double x) const {
  return arithmetic::evaluate_pdf::left_const_operation(
      x, operation, l_operand, [this](double t) { return r_operand->Pdf(t); });
};

template <typename T, typename U>
double CompoundDistribution<T, U>::Cdf(double x) const {
  return numerical_methods::cdf::compute(
      [this](double y) { return this->Pdf(y); }, x, cdf_integrator);
};

template <typename T>
double CompoundDistribution<T, double>::Cdf(double x) const {
  return numerical_methods::cdf::compute(
      [this](double y) { return this->Pdf(y); }, x, cdf_integrator);
};

template <typename U>
double CompoundDistribution<double, U>::Cdf(double x) const {
  return numerical_methods::cdf::compute(
      [this](double y) { return this->Pdf(y); }, x, cdf_integrator);
};

template <typename T, typename U>
double CompoundDistribution<T, U>::Mgf(double t) const {
  throw std::logic_error(
      "Moment generating function not implemented for CompoundDistribution");
};

template <typename T>
double CompoundDistribution<T, double>::Mgf(double x) const {
  throw std::logic_error(
      "Moment generating function not implemented for CompoundDistribution");
};

template <typename U>
double CompoundDistribution<double, U>::Mgf(double x) const {
  throw std::logic_error(
      "Moment generating function not implemented for CompoundDistribution");
};

template <typename T, typename U>
std::complex<double> CompoundDistribution<T, U>::Cf(double t) const {
  throw std::logic_error(
      "Characteristic function not implemented for CompoundDistribution");
};

template <typename T>
std::complex<double> CompoundDistribution<T, double>::Cf(double x) const {
  throw std::logic_error(
      "Characteristic function not implemented for CompoundDistribution");
};

template <typename U>
std::complex<double> CompoundDistribution<double, U>::Cf(double x) const {
  throw std::logic_error(
      "Characteristic function not implemented for CompoundDistribution");
};

template <typename T, typename U>
RV<CompoundDistribution<T, U>> operator+(RV<T> const &var_a,
                                         RV<U> const &var_b) {
  return RV<CompoundDistribution<T, U>>(CompoundDistribution(
      &(var_a.probability_distribution), &(var_b.probability_distribution),
      arithmetic::addition));
};

template <typename T, typename U>
RV<CompoundDistribution<T, U>> operator-(RV<T> const &var_a,
                                         RV<U> const &var_b) {
  return RV<CompoundDistribution<T, U>>(CompoundDistribution(
      &(var_a.probability_distribution), &(var_b.probability_distribution),
      arithmetic::subtraction));
};

template <typename T, typename U>
RV<CompoundDistribution<T, U>> operator*(RV<T> const &var_a,
                                         RV<U> const &var_b) {
  return RV<CompoundDistribution<T, U>>(CompoundDistribution(
      &(var_a.probability_distribution), &(var_b.probability_distribution),
      arithmetic::multiplication));
};

template <typename T, typename U>
RV<CompoundDistribution<T, U>> operator/(RV<T> const &var_a,
                                         RV<U> const &var_b) {
  return RV<CompoundDistribution<T, U>>(CompoundDistribution(
      &(var_a.probability_distribution), &(var_b.probability_distribution),
      arithmetic::division));
};

template <typename T, typename U>
RV<CompoundDistribution<T, U>> operator^(RV<T> const &var_a,
                                         RV<U> const &var_b) {
  return RV<CompoundDistribution<T, U>>(CompoundDistribution(
      &(var_a.probability_distribution), &(var_b.probability_distribution),
      arithmetic::exponentiation));
};

template <typename T>
RV<CompoundDistribution<T, double>> operator+(RV<T> const &var_a,
                                              double var_b) {
  return RV<CompoundDistribution<T, double>>(CompoundDistribution<T, double>(
      &(var_a.probability_distribution), var_b, arithmetic::addition));
};

template <typename U>
RV<CompoundDistribution<double, U>> operator+(double var_a,
                                              RV<U> const &var_b) {
  return RV<CompoundDistribution<double, U>>(CompoundDistribution<double, U>(
      var_a, &(var_b.probability_distribution), arithmetic::addition));
};

template <typename T>
RV<CompoundDistribution<T, double>> operator-(RV<T> const &var_a,
                                              double var_b) {
  return RV<CompoundDistribution<T, double>>(CompoundDistribution<T, double>(
      &(var_a.probability_distribution), var_b, arithmetic::subtraction));
};

template <typename U>
RV<CompoundDistribution<double, U>> operator-(double var_a,
                                              RV<U> const &var_b) {
  return RV<CompoundDistribution<double, U>>(CompoundDistribution<double, U>(
      var_a, &(var_b.probability_distribution), arithmetic::subtraction));
};

template <typename T>
RV<CompoundDistribution<T, double>> operator*(RV<T> const &var_a,
                                              double var_b) {
  return RV<CompoundDistribution<T, double>>(CompoundDistribution<T, double>(
      &(var_a.probability_distribution), var_b, arithmetic::multiplication));
};

template <typename U>
RV<CompoundDistribution<double, U>> operator*(double var_a,
                                              RV<U> const &var_b) {
  return RV<CompoundDistribution<double, U>>(CompoundDistribution<double, U>(
      var_a, &(var_b.probability_distribution), arithmetic::multiplication));
};

template <typename T>
RV<CompoundDistribution<T, double>> operator/(RV<T> const &var_a,
                                              double var_b) {
  return RV<CompoundDistribution<T, double>>(CompoundDistribution<T, double>(
      &(var_a.probability_distribution), var_b, arithmetic::division));
};

template <typename U>
RV<CompoundDistribution<double, U>> operator/(double var_a,
                                              RV<U> const &var_b) {
  return RV<CompoundDistribution<double, U>>(CompoundDistribution<double, U>(
      var_a, &(var_b.probability_distribution), arithmetic::division));
};

template <typename T>
RV<CompoundDistribution<T, double>> operator^(RV<T> const &var_a,
                                              double var_b) {
  return RV<CompoundDistribution<T, double>>(CompoundDistribution<T, double>(
      &(var_a.probability_distribution), var_b, arithmetic::exponentiation));
};

template <typename U>
RV<CompoundDistribution<double, U>> operator^(double var_a,
                                              RV<U> const &var_b) {
  return RV<CompoundDistribution<double, U>>(CompoundDistribution<double, U>(
      var_a, &(var_b.probability_distribution), arithmetic::exponentiation));
};

}  // namespace random_variable
}  // namespace shrew