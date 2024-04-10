cmake_minimum_required(VERSION 3.25.0)
project(shrew)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    gnuplot
    GIT_REPOSITORY https://github.com/gnuplot/gnuplot.git
    GIT_TAG 5.5
)

FetchContent_MakeAvailable(gnuplot)

add_executable(shrew src/test.cpp src/normal_distribution.cpp src/random_variable.cpp)