

add_library(${PROJECT_NAME} STATIC 
  numerical_methods.cpp 
  arithmetic.cpp 
  logic_assertions.cpp 
  normal_distribution.cpp 
  generic_distribution.cpp)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${boostmath_SOURCE_DIR}/include
        ${eigen_SOURCE_DIR}
)

