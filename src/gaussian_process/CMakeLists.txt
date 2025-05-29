add_library(gp
    gaussian_process.cpp
    matern_kernel.cpp
    matern_kernel_extended.cpp
    squared_exponential_kernel.cpp
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>        
)

target_link_libraries(
  gp PUBLIC
  shrew
  NLopt::nlopt
)

set(public_headers
    ${CMAKE_SOURCE_DIR}/include/shrew/gaussian_process/gaussian_process.hpp
    ${CMAKE_SOURCE_DIR}/include/shrew/gaussian_process/kernel.hpp
    ${CMAKE_SOURCE_DIR}/include/shrew/gaussian_process/matern_kernel_extended.hpp
    ${CMAKE_SOURCE_DIR}/include/shrew/gaussian_process/matern_kernel.hpp
    ${CMAKE_SOURCE_DIR}/include/shrew/gaussian_process/squared_exponential_kernel.hpp)

set_target_properties(gp PROPERTIES PUBLIC_HEADER "${public_headers}")

install(TARGETS gp
    EXPORT "${PROJECT_NAME}Targets"
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/gaussian_process
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)