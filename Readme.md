# Shrew
Static library for random variable arithmetic.

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
There are python bindings for the shrew package. To install it, you can use `pip install <shrew path>/install/pyshrew` after running the cmake installation described above. Ensure the `PYTHON_EXECUTABLE` is set to the same python instance as the pip command by running
```bash
cmake .. -DPYTHON_ECECUTABLE=<path-to-your-python-executable>
```