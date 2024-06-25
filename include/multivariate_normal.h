#pragma once

#include "random_vector.h"

namespace shrew {
namespace random_vector {

/// @brief Multivariate nomal distribution as a random vector
/// @tparam n 
template <int n>
class MultivariateNormal : public RandomVector<n> {
 public:
  virtual double joint_pdf(Eigen::Matrix<double, n, 1> x) const override;
  virtual RandomVector<n> * marginal(int marginalized_indices[]) const override;
  virtual RandomVector<n> * conditional(int conditioned_indices[], char _operator, double value[]) const override;

  MultivariateNormal(Eigen::Matrix<double, n, 1> mu, Eigen::Matrix<double, n, n> K) : K(K), mu(mu) 
    {
      det_K = K.determinant();
      K_inv = K.inverse();
    };

 private:
  const Eigen::Matrix<double, n, n> K;
  const Eigen::Matrix<double, n, 1> mu;
  double det_K;
  Eigen::Matrix<double, n, n> K_inv;
};

template<int n>
double MultivariateNormal<n>::joint_pdf(Eigen::Matrix<double, n, 1> x) const {
  return pow(2 * M_PI, -n/2.0) / sqrt(det_K) * exp(-0.5 * (x - mu).transpose() * K_inv * (x - mu));
}

template<int n>
RandomVector<n> * MultivariateNormal<n>::marginal(int marginalized_indices[]) const {
  return new MultivariateNormal<n>(mu, K);
}

template<int n>
RandomVector<n> * MultivariateNormal<n>::conditional(int conditioned_indices[], char _operator, double value[]) const {
  return new MultivariateNormal<n>(mu, K);
}

}  // namespace random_variable
}  // namespace shrew