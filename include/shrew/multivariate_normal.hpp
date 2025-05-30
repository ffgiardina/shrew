#pragma once

#include "shrew/random_vector.hpp"

#include <vector>
#include <numeric>

namespace shrew
{
  namespace random_vector
  {

    /// @brief Multivariate nomal distribution as a random vector
    /// @tparam n
    class MultivariateNormal : public RandomVector
    {
    public:
      virtual double joint_pdf(Eigen::VectorXd x) const override;

      MultivariateNormal(Eigen::VectorXd mu, Eigen::MatrixXd K) : K(K), mu(mu)
      {
        det_K = K.determinant();
        K_inv = K.inverse();
      };

      const Eigen::MatrixXd K;
      const Eigen::VectorXd mu;

    private:
      double det_K;
      Eigen::MatrixXd K_inv;
    };

    MultivariateNormal getMarginal(MultivariateNormal random_vector, Eigen::VectorXi marginal_indices);
    MultivariateNormal getConditional(MultivariateNormal random_vector, Eigen::VectorXi conditional_indices, char _operator, Eigen::MatrixXd value);
    MultivariateNormal getConditional(MultivariateNormal random_vector, std::tuple<int, int> conditional_index_range, char _operator, Eigen::VectorXd value);
  } // namespace random_variable
} // namespace shrew