#pragma once

#include <vector>

#include "random_variable.h"

namespace shrew {
namespace random_variable {

class NormalDistribution : ProbabilityDistribution<double> {
 public:
  // Probability density function
  virtual double Pdf(std::vector<double> x) override;

  // Cumulative distribution function
  virtual double Cdf(std::vector<double> x) override;

  // Moment generating function
  virtual double Mgf(std::vector<double> x) override;

  // Characteristic function
  virtual double Cf(std::vector<double> x) override;  
};

}  // namespace random_variable
}  // namespace shrew