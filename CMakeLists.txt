cmake_minimum_required(VERSION 3.25.0)
project(shrew)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MINGW)
    add_compile_options(-Wa,-mbig-obj)
endif()

list(APPEND CMAKE_PREFIX_PATH $ENV{SHREW_DEPS})

find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)

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
        src/generic_distribution.cpp)

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

enable_testing()
add_subdirectory("tests")
