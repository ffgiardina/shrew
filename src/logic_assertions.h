#include "random_variable.h"

#include <tuple>
#include <unordered_set>

namespace shrew
{
    namespace random_variable
    {
        bool has_repeating_random_variable(ProbabilityDistribution const *var, std::unordered_set<const ProbabilityDistribution*> &vars);
    } // namespace random_variable
} // namespace shrew