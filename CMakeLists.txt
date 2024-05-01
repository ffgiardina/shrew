cmake_minimum_required(VERSION 3.25.0)
project(shrew)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
include(FetchContent)
FetchContent_Declare(
  boostmath
  GIT_REPOSITORY https://github.com/boostorg/math.git
  GIT_TAG boost-1.84.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(
  googletest
  boostmath)

# Create library
add_library(${PROJECT_NAME} STATIC src/numerical_methods.cpp src/arithmetic.cpp src/normal_distribution.cpp src/generic_distribution.cpp src/random_variable.cpp)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${boostmath_SOURCE_DIR}/include
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

# Test
enable_testing()

message(PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
message(PROJECT_SOURCE_DIR="${boostmath_SOURCE_DIR}")

add_executable(shrew_tests tests/normal_distribution.cpp tests/generic_distribution.cpp)
target_link_libraries(
  shrew_tests
  GTest::gtest_main
  Boost::math
  shrew
)

get_target_property(OUT Boost::math LINK_LIBRARIES)
message(STATUS ${OUT})

include(GoogleTest)
gtest_discover_tests(shrew_tests)