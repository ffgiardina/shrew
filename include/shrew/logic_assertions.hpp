#pragma once

#include <tuple>
#include <unordered_set>

#include "shrew/random_variable.hpp"

namespace shrew
{
    namespace random_variable
    {
        class LogicAssertions
        {
        public:
            static bool has_repeating_random_variable(
                ProbabilityDistribution const *var,
                std::unordered_set<const ProbabilityDistribution *> &vars);
        };

    } // namespace random_variable
} // namespace shrew