#include <math.h>
#include "normal_distribution.h"

namespace shrew {
namespace random_variable {

double NormalDistribution::Pdf(double x) 
{
    return 1.0 / (sigma * sqrt(2 * M_PI)) * exp(- pow(x - mu, 2) / (2 * pow(sigma, 2)));
};

double NormalDistribution::Cdf(double x) 
{
    return x;
};

double NormalDistribution::Mgf(double x) 
{
    return x;
};

double NormalDistribution::Cf(double x) 
{
    return x;
};

}  // namespace random_variable
}  // namespace shrew