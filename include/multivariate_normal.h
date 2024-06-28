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

  MultivariateNormal(Eigen::Matrix<double, n, 1> mu, Eigen::Matrix<double, n, n> K) : K(K), mu(mu) 
    {
      det_K = K.determinant();
      K_inv = K.inverse();
    };

  const Eigen::Matrix<double, n, n> K;
  const Eigen::Matrix<double, n, 1> mu;

 private:
  double det_K;
  Eigen::Matrix<double, n, n> K_inv;

};

template<int n>
double MultivariateNormal<n>::joint_pdf(Eigen::Matrix<double, n, 1> x) const {
  return pow(2 * M_PI, -n/2.0) / sqrt(det_K) * exp(-0.5 * (x - mu).transpose() * K_inv * (x - mu));
}

template<int n, int m>
MultivariateNormal<m> getMarginal(MultivariateNormal<n> random_vector, std::vector<int> marginal_indices) {
  return MultivariateNormal<m>(random_vector.mu(marginal_indices), random_vector.K(marginal_indices, marginal_indices));
}

template<int n, int m>
MultivariateNormal<m> getConditional(std::vector<int> conditioned_indices[], char _operator, std::vector<double> value) {
  return new MultivariateNormal<m>();
}

}  // namespace random_variable
}  // namespace shrew