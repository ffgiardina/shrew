#include "shrew/multivariate_normal.hpp"

#include <vector>
#include <numeric>
#include <unordered_set>

namespace shrew
{
  namespace random_vector
  {
    double MultivariateNormal::joint_pdf(Eigen::VectorXd x) const
    {
      auto exponent = -0.5 * (x - mu).transpose() * K_inv * (x - mu);
      return pow(2 * M_PI, -x.size() / 2.0) / sqrt(det_K) * exp(exponent);
    }

    MultivariateNormal getMarginal(MultivariateNormal random_vector, Eigen::VectorXi marginal_indices)
    {
      return MultivariateNormal(random_vector.mu(marginal_indices), random_vector.K(marginal_indices, marginal_indices));
    }

    MultivariateNormal getConditional(MultivariateNormal random_vector, Eigen::VectorXi conditional_indices, char _operator, Eigen::MatrixXd value)
    {
      auto conditional_index_set = std::unordered_set<int>(conditional_indices.begin(), conditional_indices.end());
      int n = random_vector.mu.size();
      Eigen::VectorXi non_conditional_indices(n - conditional_indices.size());
      for (int i = 0, k = 0; i < n; ++i) {
        if (!conditional_index_set.contains(i)) {
          non_conditional_indices[k] = i;
          ++k;
        }
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
      Eigen::VectorXi conditional_indices(std::max(0, delta));
      std::iota(std::begin(conditional_indices), std::end(conditional_indices), std::get<0>(conditional_index_range));
      return getConditional(random_vector, conditional_indices, _operator, value);
    }
  } // namespace random_variable
} // namespace shrew
