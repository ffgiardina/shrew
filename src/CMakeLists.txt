include(FetchContent)
FetchContent_Declare(
  boostmath
  GIT_REPOSITORY https://github.com/boostorg/math.git
  GIT_TAG boost-1.84.0
)

FetchContent_MakeAvailable(boostmath)

add_library(${PROJECT_NAME} STATIC numerical_methods.cpp arithmetic.cpp logic_assertions.cpp normal_distribution.cpp generic_distribution.cpp)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${boostmath_SOURCE_DIR}/include
)