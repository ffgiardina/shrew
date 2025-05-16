cmake_minimum_required(VERSION 3.25.0)
cmake_policy(SET CMP0148 OLD) # Make sure provided python executable is used
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
project(shrew)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH $ENV{SHREW_DEPS})

find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(NLopt REQUIRED)
find_package(pybind11 REQUIRED)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/install" CACHE PATH "" FORCE)
endif()
message(
    STATUS
    "CMAKE_INSTALL_PREFIX set to ${CMAKE_INSTALL_PREFIX}\n"
)

add_library(${PROJECT_NAME} STATIC)

target_sources(${PROJECT_NAME}
    PRIVATE
        src/numerical_methods.cpp 
        src/arithmetic.cpp 
        src/logic_assertions.cpp 
        src/normal_distribution.cpp 
        src/generic_distribution.cpp
        src/multivariate_normal.cpp
        src/gaussian_process/gaussian_process.cpp
        src/gaussian_process/maternKernel.cpp
        src/gaussian_process/maternKernelNoise.cpp
        src/gaussian_process/squaredExponentialKernel.cpp)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>        
)

target_link_libraries(
  shrew
  Boost::boost
  Eigen3::Eigen
  NLopt::nlopt
)

set(public_headers
    include/shrew/compound_distribution.hpp
    include/shrew/delta_distribution.hpp
    include/shrew/generic_distribution.hpp
    include/shrew/multivariate_normal.hpp
    include/shrew/normal_distribution.hpp
    include/shrew/random_variable.hpp
    include/shrew/random_vector.hpp
    include/shrew/arithmetic.hpp
    include/shrew/logic_assertions.hpp
    include/shrew/numerical_methods.hpp
    include/shrew/gaussian_process/gaussian_process.hpp
    include/shrew/gaussian_process/kernel.hpp
    include/shrew/gaussian_process/maternKernel.hpp
    include/shrew/gaussian_process/maternKernelNoise.hpp
    include/shrew/gaussian_process/squaredExponentialKernel.hpp)

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${public_headers}")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "d")

pybind11_add_module(pyshrew
    bindings/pyshrew_bind.cpp
    bindings/bind_distributions.cpp
    bindings/bind_random_variable.cpp
    bindings/bind_random_vector.cpp
)
target_link_libraries(pyshrew PRIVATE shrew)

# Installation
include(GNUInstallDirs) # Define GNU standard installation directories
install(TARGETS ${PROJECT_NAME}
    EXPORT "${PROJECT_NAME}Targets"
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT "${PROJECT_NAME}Targets"
    FILE "${PROJECT_NAME}Targets.cmake"
    NAMESPACE Shrew::
    DESTINATION cmake
)

include(CMakePackageConfigHelpers)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION cmake
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    DESTINATION cmake
)

# Python pip-compatible structure installation
install(TARGETS pyshrew
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/pyshrew/pyshrew
)

# Install __init__.py so pip sees it as a package
install(FILES bindings/__init__.py
    DESTINATION ${CMAKE_INSTALL_PREFIX}/pyshrew/pyshrew
)

# Optional: Install other metadata
install(FILES bindings/setup.py
    DESTINATION ${CMAKE_INSTALL_PREFIX}/pyshrew
)

enable_testing()
add_subdirectory("tests")
