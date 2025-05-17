# Add source files
set(MLCPP_SOURCES
    matrix.cpp
    linear_regression.cpp
    logistic_regression.cpp
    neural_network.cpp
    dataset.cpp
)

# Create library target
add_library(mlcpp_core ${MLCPP_SOURCES})

# Set include directories
target_include_directories(mlcpp_core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link dependencies
target_link_libraries(mlcpp_core
    PUBLIC
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
)

# Set compile options
target_compile_options(mlcpp_core
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic -march=native>
        $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic -march=native>
        $<$<BOOL:${MLCPP_ENABLE_SIMD}>:-mavx2 -mfma>
)

# CUDA support
if(MLCPP_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    target_compile_definitions(mlcpp_core PRIVATE MLCPP_CUDA_ENABLED)
    target_include_directories(mlcpp_core PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(mlcpp_core PRIVATE ${CUDA_LIBRARIES})
endif() 