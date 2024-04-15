#pragma once

#include <complex>

namespace shrew {
namespace random_variable {

class ProbabilityDistribution {
 public:
  // Probability density function
  virtual double Pdf(double x) = 0;

  // Cumulative distribution function
  virtual double Cdf(double x) = 0;

  // Moment generating function
  virtual double Mgf(double t) = 0;

  // Characteristic function
  virtual std::complex<double> Cf(double t) = 0;  
};

template<typename T>
class RandomVariable {
 public:
  T probability_distribution;
  void Add();
  void Subtract();
  RandomVariable(T pdist) {
    probability_distribution = pdist;
  };
  RandomVariable<T> operator+(RandomVariable<T> const &var);
  RandomVariable<T> operator-(RandomVariable<T> const &var);
};

template<typename T, int n>
class RandomVector {
 public:
  RandomVariable<T> vector;
  void Add();
  void Subtract();
  
  RandomVector() {
  };
};

}  // namespace random_variable
}  // namespace shrew