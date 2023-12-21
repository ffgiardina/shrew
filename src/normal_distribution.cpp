#include <vector>

#include "normal_distribution.h"

namespace shrew {
namespace random_variable {

double NormalDistribution::Pdf(std::vector<double> x) 
{
    return x.at(0);
};

double NormalDistribution::Cdf(std::vector<double> x) 
{
    return x.at(0);
};

double NormalDistribution::Mgf(std::vector<double> x) 
{
    return x.at(0);
};

double NormalDistribution::Cf(std::vector<double> x) 
{
    return x.at(0);
};

}  // namespace random_variable
}  // namespace shrew