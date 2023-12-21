#pragma once

#include <vector>

namespace shrew {
namespace random_variable {

template<typename T>
class ProbabilityDistribution {
 public:
  // Probability density function
  virtual double Pdf(std::vector<T> x) = 0;

  // Cumulative distribution function
  virtual double Cdf(std::vector<T> x) = 0;

  // Moment generating function
  virtual double Mgf(std::vector<T> x) = 0;

  // Characteristic function
  virtual double Cf(std::vector<T> x) = 0;  
};

template<typename T>
class RandomVariable : ProbabilityDistribution<T> {
 public:
  void Add();
  void Subtract();
  void Multiply();
  void Divide();
  void Power();
  void Logarithm();
};

}  // namespace random_variable
}  // namespace shrew