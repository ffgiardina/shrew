#include "random_variable.h"

#include <tuple>
#include <unordered_set>

namespace shrew
{
    namespace random_variable
    {
        bool has_repeating_random_variable(ProbabilityDistribution const *var, std::unordered_set<const ProbabilityDistribution*> &vars)
        {
            if (!var)
                return false;
            if (vars.contains(var))
                return true;
            
            vars.insert(var);
            auto [l_operand, r_operand] = var->get_operands();
            return has_repeating_random_variable(l_operand, vars) || has_repeating_random_variable(r_operand, vars);
        };
    } // namespace random_variable
} // namespace shrew