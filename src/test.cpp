#include<vector>
#include<iostream>

#include "normal_distribution.h"

using namespace shrew::random_variable;

int main()
{
    NormalDistribution normal = NormalDistribution(0, 1);
    // std::cout << normal.mu << normal.sigma << std::endl;
    RandomVariable<NormalDistribution> grv = RandomVariable<NormalDistribution>(normal);
    
    int n {10};
    double y[n];
    for (int i {0}; i<n; ++i) {
        y[i] = grv.Evaluate(i);
        std::cout << y[i] << " ";
    }
    return 0;
};