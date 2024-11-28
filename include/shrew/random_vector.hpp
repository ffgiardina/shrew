#pragma once

#include <Eigen/Dense>

namespace shrew
{
  namespace random_vector
  {

    /// @brief Base class of a random vector
    /// @tparam n
    class RandomVector
    {
    public:
      virtual double joint_pdf(Eigen::VectorXd x) const = 0;
    };

  } // namespace random_variable
} // namespace shrew