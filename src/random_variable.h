#pragma once

namespace shrew {
namespace random_variable {

class ProbabilityDistribution {
 public:
  // Probability density function
  virtual double Pdf(double x) = 0;

  // Cumulative distribution function
  virtual double Cdf(double x) = 0;

  // Moment generating function
  virtual double Mgf(double x) = 0;

  // Characteristic function
  virtual double Cf(double x) = 0;  
};

template<typename T>
class RandomVariable {
 public:
  T probability_distribution;
  void Add();
  void Subtract();
  double Evaluate(double x) {
    return probability_distribution.Pdf(x);
  };
  RandomVariable(T pdist) {
    probability_distribution = pdist;
  };
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