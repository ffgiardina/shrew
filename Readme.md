# Shrew
Static C++ library with python bindings for random variable arithmetic. 

Currently featured:

- Arithmetic of normal random variables (e.g. X * Y - Z)
- Probability density and cumulative density of arithmetic expressions with lazy numerical evaluation if no exact solution exists
- Conditionals and marginals of multivariate normal random vectors

## Requirements
Shrew requires the `Eigen C++` and `Boost.Math` libraries. If these dependencies are not installed in the default paths, set the environment variable `SHREW_DEPS` to the root directory where they are located.

## Installation
### CMake
After cloning the remote repository, open the terminal and set the working directory to the local repository
```console
cd <PathToRepo>/shrew
mkdir build
cd build
cmake ..
cmake --build .
cmake --install .
```
The installed library should now be in the `install` directory inside the repository. 
To use it in other projects, import it using the `find_package(shrew)` command in your `CMakeLists.txt`. You may have to append the install directory to your `CMAKE_PREFIX_PATH` variable. Finally, add the library to your target using `target_link_libraries` with the shrew library target name `Shrew::shrew`.

### Shrew python package
There are python bindings for the shrew package. Currently only support for binary arithmetic operations of random variables and vectors of normal distributions and operations of normal random variables with constants. 

To install it, you can use `pip install <shrew path>/install/pyshrew` after running the cmake installation described above. Ensure the `PYTHON_EXECUTABLE` is set to the same python instance as the pip command by running
```bash
cmake .. -DPYTHON_EXECUTABLE=<path-to-your-python-executable>
```

## Python Examples

### Random variable arithmetic
```python
import pyshrew as ps

rv1 = ps.RandomVariable(ps.NormalDistribution(mean=-1.0, stddev=1.0))
rv2 = ps.RandomVariable(ps.NormalDistribution(mean=1.0, stddev=1.0))

# Cumulative distribution of compound random variable at 0.0
print((rv1 * rv2).pdist.cdf(0.0))

# Probability density distribution of compound random variable at 2.0
print((0.5 / rv1).pdist.cdf(2.0))
```

### Random vector operations
```python
import pyshrew as ps
import numpy as np

mu = np.array([1, 2])  # Mean
K = np.array([[4, 1.5], [1.5, 1]])  # Covariance matrix

rvec = ps.RandomVector(mu, K)

# Define conditional indices, condition type, and condition values
# below, we define the 0th index to be equal (=) to 3
indices = [0]
condition = '='
values = np.array([3])

# Get conditional random vector
conditional_rvec = ps.get_conditional(rvec, indices, condition, values)

print(f'Conditional mean: {conditional_rvec.mu}')
print(f'Conditional variance: {conditional_rvec.K}')
```
