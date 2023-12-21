#include<vector>
#include<iostream>

#include "normal_distribution.h"

using namespace shrew::random_variable;

int main()
{
    NormalDistribution grv = NormalDistribution();
    std::vector<double> vect(0, 1);
    std::cout << grv.Pdf(vect);
    return 0;
};