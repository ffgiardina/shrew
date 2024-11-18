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
    template <int n>
    class MultivariateNormal : public RandomVector<n>
    {
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

    template <>
    double MultivariateNormal<1>::joint_pdf(Eigen::Matrix<double, 1, 1> x) const
    {
      auto exponent = -0.5 * (x - mu).transpose() * K_inv * (x - mu);
      return pow(2 * M_PI, -1.0 / 2.0) / sqrt(det_K) * exp(exponent.value());
    }

    template <int n>
    double MultivariateNormal<n>::joint_pdf(Eigen::Matrix<double, n, 1> x) const
    {
      auto exponent = -0.5 * (x - mu).transpose() * K_inv * (x - mu);
      return pow(2 * M_PI, -n / 2.0) / sqrt(det_K) * exp(exponent);
    }

    template <int n, int m>
    MultivariateNormal<m> getMarginal(MultivariateNormal<n> random_vector, std::vector<int> marginal_indices)
    {
      return MultivariateNormal<m>(random_vector.mu(marginal_indices), random_vector.K(marginal_indices, marginal_indices));
    }

    template <int n, int m>
    MultivariateNormal<m> getConditional(MultivariateNormal<n> random_vector, std::vector<int> conditional_indices, char _operator, Eigen::Matrix<double, n - m, 1> value)
    {
      std::vector<int> non_conditional_indices(n - conditional_indices.size());
      for (int i = 0, k = 0; i < n; ++i)
        if (i == conditional_indices[k])
          k += 1;
        else
          non_conditional_indices[i-k] = i;

      Eigen::Matrix<double, m, 1> mu_conditioned;
      Eigen::Matrix<double, m, m> K_conditioned;
      if (_operator == '=')
      {
        auto mu_1 = random_vector.mu(non_conditional_indices);
        auto mu_2 = random_vector.mu(conditional_indices);
        auto sigma_11 = random_vector.K(non_conditional_indices, non_conditional_indices);
        auto sigma_12 = random_vector.K(non_conditional_indices, conditional_indices);
        auto sigma_22_inv = random_vector.K(conditional_indices, conditional_indices).inverse();

        mu_conditioned = mu_1 + sigma_12 * sigma_22_inv * (value - mu_2);
        K_conditioned = sigma_11 - sigma_12 * sigma_22_inv * sigma_12.transpose();
      }
      else
      {
        throw std::logic_error("Error: Requested conditional operator not implemented for multivariate normal distribution.");
      }

      return MultivariateNormal<m>(mu_conditioned, K_conditioned);
    }

    template <int n, int m>
    MultivariateNormal<m> getConditional(MultivariateNormal<n> random_vector, std::tuple<int, int> conditional_index_range, char _operator, Eigen::Matrix<double, n - m, 1> value)
    {
      auto delta = std::get<1>(conditional_index_range) - std::get<0>(conditional_index_range) + 1;
      if (delta <= 0)
        throw std::logic_error("Error: Start index greater than end index for conditional range.");
      std::vector<int> conditional_indices(std::max(0, delta));
      std::iota(std::begin(conditional_indices), std::end(conditional_indices), std::get<0>(conditional_index_range));
      return getConditional<n, m>(random_vector, conditional_indices, _operator, value);
    }
  } // namespace random_variable
} // namespace shrew