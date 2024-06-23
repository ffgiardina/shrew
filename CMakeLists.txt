cmake_minimum_required(VERSION 3.25.0)
project(shrew)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  boostmath
  GIT_REPOSITORY https://github.com/boostorg/math.git
  GIT_TAG boost-1.84.0
)

include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)

FetchContent_MakeAvailable(boostmath)
FetchContent_MakeAvailable(eigen)

include_directories(${eigen_SOURCE_DIR})

add_subdirectory("src")
add_subdirectory("tests")