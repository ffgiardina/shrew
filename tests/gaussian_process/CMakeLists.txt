include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

set(INSTALL_GMOCK OFF)
set(INSTALL_GTEST OFF)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(shrew_tests_gp gaussian_process.cpp)
target_link_libraries(
  shrew_tests_gp
  GTest::gtest_main
  shrew
  gp
)

add_test(NAME SHREWTESTS_GP COMMAND shrew_tests_gp)

include(GoogleTest)
