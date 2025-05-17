cmake_minimum_required(VERSION 3.15)
project(mlcpp VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable testing
enable_testing()

# Find required packages
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)

# Options
option(MLCPP_BUILD_TESTS "Build tests" ON)
option(MLCPP_ENABLE_CUDA "Enable CUDA support" OFF)
option(MLCPP_ENABLE_SIMD "Enable SIMD optimizations" ON)

# Add compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -march=native)
    if(MLCPP_ENABLE_SIMD)
        add_compile_options(-mavx2 -mfma)
    endif()
endif()

# Add subdirectories
add_subdirectory(src)
if(MLCPP_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Create main library target
add_library(mlcpp
    src/matrix.cpp
    src/linear_regression.cpp
    src/logistic_regression.cpp
    src/neural_network.cpp
    src/dataset.cpp
    src/optimizer.cpp
)

target_include_directories(mlcpp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(mlcpp
    PUBLIC
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
)

# Install rules
install(TARGETS mlcpp
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
) 