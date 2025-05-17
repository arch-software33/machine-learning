include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(mlcpp_tests
    test_matrix.cpp
)

target_link_libraries(mlcpp_tests
    PRIVATE
        mlcpp
        GTest::gtest_main
        GTest::gmock_main
)

include(GoogleTest)
gtest_discover_tests(mlcpp_tests) 