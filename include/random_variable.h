#pragma once

#include <complex>
#include <tuple>

namespace shrew {
namespace random_variable {

/// @brief Abstract base class of a probability distribution
class ProbabilityDistribution {
 public:
  // Probability density function
  virtual double Pdf(double x) const = 0;

  // Cumulative distribution function
  virtual double Cdf(double x) const = 0;

  // Moment generating function
  virtual double Mgf(double t) const = 0;

  // Characteristic function
  virtual std::complex<double> Cf(double t) const = 0;

 private:
  virtual std::tuple<const ProbabilityDistribution *,
                     const ProbabilityDistribution *>
  get_pd_operands() const {
    return {0, 0};
  };

  friend class LogicAssertions;
};

/// @brief Base class of a random variable
/// @tparam T
template <typename T>
class RandomVariable {
 public:
  T probability_distribution;
  RandomVariable(T pdist) { probability_distribution = pdist; };
};

}  // namespace random_variable
}  // namespace shrew