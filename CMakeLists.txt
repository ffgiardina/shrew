cmake_minimum_required(VERSION 3.25.0)
if(NOT DEFINED SKBUILD)
  cmake_policy(SET CMP0148 OLD) # Legacy Python discovery for non-scikit-build builds
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
project(shrew)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option (SHREW_PYTHON "build python bindings" ON)
option (SHREW_GP "build gaussian process subpackage" ON)
option (SHREW_TESTS "build gtests" ON)

include(FetchContent)

find_package(Eigen3 3.4 QUIET)
if(NOT TARGET Eigen3::Eigen)
  FetchContent_Declare(
    Eigen3
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    DOWNLOAD_EXTRACT_TIMESTAMP OFF
  )
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(Eigen3)
endif()
find_package(Boost REQUIRED)

if (SHREW_GP)
    find_package(NLopt QUIET)
    if(NOT TARGET NLopt::nlopt)
        find_package(PkgConfig QUIET)
        if(PKG_CONFIG_FOUND)
            pkg_check_modules(NLOPT QUIET nlopt)
            if(NLOPT_FOUND)
                add_library(NLopt::nlopt IMPORTED UNKNOWN GLOBAL)
                set_target_properties(NLopt::nlopt PROPERTIES
                    IMPORTED_LOCATION             "${NLOPT_LINK_LIBRARIES}"
                    INTERFACE_INCLUDE_DIRECTORIES "${NLOPT_INCLUDE_DIRS}"
                    INTERFACE_LINK_DIRECTORIES    "${NLOPT_LIBRARY_DIRS}")
            endif()
        endif()
    endif()
    if(NOT TARGET NLopt::nlopt)
        FetchContent_Declare(
            nlopt_src
            URL https://github.com/stevengj/nlopt/archive/refs/tags/v2.10.1.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP OFF
        )
        set(NLOPT_PYTHON OFF CACHE BOOL "" FORCE)
        set(NLOPT_MATLAB OFF CACHE BOOL "" FORCE)
        set(NLOPT_GUILE OFF CACHE BOOL "" FORCE)
        set(NLOPT_SWIG OFF CACHE BOOL "" FORCE)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(nlopt_src)
        add_library(NLopt::nlopt ALIAS nlopt)
    endif()
endif()

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
        src/multivariate_normal.cpp)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>        
)

target_link_libraries(
  shrew
  Boost::boost
  Eigen3::Eigen
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
    include/shrew/numerical_methods.hpp)

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${public_headers}")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "d")

if (SHREW_GP)
    message(STATUS "Building gaussian process subpackage")
    add_subdirectory(src/gaussian_process)
endif()

if (SHREW_PYTHON)
    message(STATUS "Building Python bindings")
    find_package(pybind11 CONFIG REQUIRED)
    if (SHREW_GP)
        pybind11_add_module(pyshrew
            bindings/pyshrew_bind.cpp
            bindings/bind_distributions.cpp
            bindings/bind_random_variable.cpp
            bindings/bind_random_vector.cpp
            bindings/bind_gaussian_process.cpp
        )
        target_link_libraries(pyshrew PRIVATE shrew gp)
    else()
        pybind11_add_module(pyshrew
            bindings/pyshrew_bind.cpp
            bindings/bind_distributions.cpp
            bindings/bind_random_variable.cpp
            bindings/bind_random_vector.cpp
        )
        target_link_libraries(pyshrew PRIVATE shrew)
    endif()
endif()

if (SHREW_TESTS)
    message(STATUS "Building tests")
    enable_testing()
    add_subdirectory(tests)
    if (SHREW_GP)
        add_subdirectory(tests/gaussian_process)
    endif()
endif()

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

if (SHREW_PYTHON)
  if(DEFINED SKBUILD)
    # scikit-build-core: install extension + __init__.py to pyshrew/
    # (maps to site-packages/pyshrew/ in the final wheel)
    install(TARGETS pyshrew LIBRARY DESTINATION pyshrew)
    install(FILES bindings/__init__.py DESTINATION pyshrew)
  else()
    # Legacy direct-cmake install
    message(STATUS "Installing Python bindings")
    install(TARGETS pyshrew
      LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/pyshrew/pyshrew
    )
    install(FILES bindings/__init__.py
      DESTINATION ${CMAKE_INSTALL_PREFIX}/pyshrew/pyshrew
    )
  endif()
endif()
