#pragma once

#include "random_variable.h"
#include <functional>
#include <complex>
#include <memory>

namespace shrew {
namespace numerical_methods {

/// @brief Abstract interface for a numerical integrator
class Integrator 
{
  public:
   // Function that maps the integration domain to [0, 1]
   virtual std::function<double(double)> MapDomain(std::function<double(double)> map) = 0;

   // Computes the integration result over the mapped domain
   virtual double Integrate(std::function<double(double)> feval) = 0;
};

class InfiniteDomainGaussKronrod : public Integrator
{
  public:
   virtual std::function<double(double)> MapDomain(std::function<double(double)> map) override;
   virtual double Integrate(std::function<double(double)> feval) override;
   
   static const unsigned int n_point = 200;
};

class SemiInfiniteGaussKronrod : public Integrator
{
  public:
   virtual std::function<double(double)> MapDomain(std::function<double(double)> map) override;
   virtual double Integrate(std::function<double(double)> feval) override;
   
   double upper_bound;
   static const unsigned int n_point = 200;
   SemiInfiniteGaussKronrod(double upper_bound): upper_bound(upper_bound) {};
};
  
}  // namespace numerical_methods
}  // namespace shrew