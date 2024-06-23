#pragma once

#include <Eigen/Dense>

namespace shrew {
namespace random_vector {

/// @brief Base class of a random vector
/// @tparam n 
template <int n>
class RandomVector {
 public:
  virtual double joint_pdf(Eigen::Matrix<double, n, 1> x) const = 0;
  virtual RandomVector * marginal(int marginalized_indices[]) const = 0;
  virtual RandomVector * conditional(int conditioned_indices[], char _operator, double value[]) const = 0;
};

}  // namespace random_variable
}  // namespace shrew