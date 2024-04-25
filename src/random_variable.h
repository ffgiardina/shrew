#pragma once

#include <complex>

namespace shrew {
namespace random_variable {

/// @brief Abstract base class of a probability distribution
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

/// @brief Base class of a random variable
/// @tparam T 
template<typename T>
class RandomVariable {
 public:
  T probability_distribution;
  RandomVariable(T pdist) {
    probability_distribution = pdist;
  };
};

/// @brief Base class of a random vector
/// @tparam T 
/// @tparam n 
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