#include "shrew/multivariate_normal.hpp"

#include <vector>
#include <numeric>

namespace shrew
{
  namespace random_vector
  {
    double MultivariateNormal::joint_pdf(Eigen::VectorXd x) const
    {
      auto exponent = -0.5 * (x - mu).transpose() * K_inv * (x - mu);
      return pow(2 * M_PI, -x.size() / 2.0) / sqrt(det_K) * exp(exponent);
    }

    MultivariateNormal getMarginal(MultivariateNormal random_vector, std::vector<int> marginal_indices)
    {
      return MultivariateNormal(random_vector.mu(marginal_indices), random_vector.K(marginal_indices, marginal_indices));
    }

    MultivariateNormal getConditional(MultivariateNormal random_vector, std::vector<int> conditional_indices, char _operator, Eigen::MatrixXd value)
    {
      int n = random_vector.mu.size();
      std::vector<int> non_conditional_indices(n - conditional_indices.size());
      for (int i = 0, k = 0; i < n; ++i) {
        auto c_index = conditional_indices[k];
        if (c_index < 0 || c_index >= n)
          throw std::out_of_range("Conditional index out of bounds");
        if (k < conditional_indices.size() && i == c_index)
          k += 1;
        else
          non_conditional_indices[i-k] = i;
      }

      Eigen::VectorXd mu_conditioned;
      Eigen::MatrixXd K_conditioned;
      if (_operator == '=')
      {
        auto mu_1 = random_vector.mu(non_conditional_indices);
        auto mu_2 = random_vector.mu(conditional_indices);
        auto sigma_11 = random_vector.K(non_conditional_indices, non_conditional_indices);
        auto sigma_12 = random_vector.K(non_conditional_indices, conditional_indices);
        auto sigma_22 = random_vector.K(conditional_indices, conditional_indices);

        Eigen::LLT<Eigen::MatrixXd> cholesky_K(sigma_22);
        Eigen::MatrixXd L = cholesky_K.matrixL();
        Eigen::VectorXd alpha = L.transpose().fullPivHouseholderQr().solve(L.fullPivHouseholderQr().solve(value - mu_2));
        Eigen::MatrixXd v = L.fullPivHouseholderQr().solve(sigma_12.transpose());

        mu_conditioned = mu_1 + sigma_12 * alpha;
        K_conditioned = sigma_11 - v.transpose() * v;
      }
      else
      {
        throw std::logic_error("Error: Requested conditional operator not implemented for multivariate normal distribution.");
      }

      return MultivariateNormal(mu_conditioned, K_conditioned);
    }

    MultivariateNormal getConditional(MultivariateNormal random_vector, std::tuple<int, int> conditional_index_range, char _operator, Eigen::VectorXd value)
    {
      auto delta = std::get<1>(conditional_index_range) - std::get<0>(conditional_index_range) + 1;
      if (delta <= 0)
        throw std::logic_error("Error: Start index greater than end index for conditional range.");
      std::vector<int> conditional_indices(std::max(0, delta));
      std::iota(std::begin(conditional_indices), std::end(conditional_indices), std::get<0>(conditional_index_range));
      return getConditional(random_vector, conditional_indices, _operator, value);
    }
  } // namespace random_variable
} // namespace shrew
