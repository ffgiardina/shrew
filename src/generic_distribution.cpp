#include <math.h>
#include "generic_distribution.h"
#include <functional>
#include <cmath>
#include <complex>

namespace shrew {
namespace random_variable {

double GenericDistribution::Pdf(double x) 
{
    return arithmetic::evaluate_pdf(x, operation, [this](double t) {return l_operand->Pdf(t); }, [this](double t) {return r_operand->Pdf(t); });
};

double GenericDistribution::Cdf(double x) 
{
    return 1.0;
};

double GenericDistribution::Mgf(double t) 
{
    return 1.0;
};

std::complex<double> GenericDistribution::Cf(double t) 
{
    std::complex<double> c(1.0, 0.0);
    return c;
};

RandomVariable<GenericDistribution> operator+(RandomVariable<GenericDistribution> const &var_a, RandomVariable<GenericDistribution> const &var_b)
{
    return GenericDistribution(std::make_shared<GenericDistribution>(var_a.probability_distribution), std::make_shared<GenericDistribution>(var_b.probability_distribution), arithmetic::addition);
};
RandomVariable<GenericDistribution> operator-(RandomVariable<GenericDistribution> const &var_a, RandomVariable<GenericDistribution> const &var_b)
{
    return GenericDistribution(std::make_shared<GenericDistribution>(var_a.probability_distribution), std::make_shared<GenericDistribution>(var_b.probability_distribution), arithmetic::subtraction);
};


}  // namespace random_variable
}  // namespace shrew