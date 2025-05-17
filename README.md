# MLCPP - A Modern C++ Machine Learning Library

MLCPP is a high-performance machine learning library implemented in modern C++. It provides implementations of common machine learning algorithms with SIMD optimizations and optional CUDA support.

## Features

- Matrix operations with SIMD optimizations
- Linear Regression
- Logistic Regression
- Neural Networks with customizable architectures
- Dataset loading and preprocessing utilities
- Comprehensive unit tests
- Modern CMake build system
- Eigen integration for advanced linear algebra operations

## Requirements

- C++20 compatible compiler
- CMake 3.15 or higher
- OpenMP
- Eigen3
- (Optional) CUDA Toolkit for GPU acceleration

## Building

```bash
mkdir build
cd build
cmake ..
make
```

To enable CUDA support:
```bash
cmake -DMLCPP_ENABLE_CUDA=ON ..
```

## Usage Examples

### Linear Regression

```cpp
#include <mlcpp/linear_regression.hpp>
#include <mlcpp/dataset.hpp>

using namespace mlcpp;

// Load data
auto [X, y] = Dataset::load_csv("data.csv", {0, 1, 2}, {3});

// Split data
auto [X_train, X_test, y_train, y_test] = Dataset::train_test_split(X, y);

// Create and train model
LinearRegression model;
model.fit(X_train, y_train);

// Make predictions
Matrix predictions = model.predict(X_test);

// Evaluate model
double score = model.score(X_test, y_test);
```

### Neural Network

```cpp
#include <mlcpp/neural_network.hpp>

// Create a neural network
NeuralNetwork nn(0.01);  // learning rate = 0.01

// Add layers
nn.add_layer(input_size, 64, Activation::ReLU);
nn.add_layer(64, 32, Activation::ReLU);
nn.add_layer(32, output_size, Activation::Sigmoid);

// Train the network
nn.fit(X_train, y_train, epochs=100, batch_size=32);

// Make predictions
Matrix predictions = nn.predict(X_test);
```

## Matrix Operations

The library provides a powerful Matrix class with SIMD-optimized operations:

```cpp
#include <mlcpp/matrix.hpp>

// Create matrices
Matrix A(100, 100);
Matrix B(100, 100);

// Perform operations
Matrix C = A + B;  // Addition
Matrix D = A * B;  // Matrix multiplication
Matrix E = A.hadamard(B);  // Element-wise multiplication
Matrix F = A.transpose();  // Transpose
```

## Dataset Utilities

```cpp
#include <mlcpp/dataset.hpp>

// Load CSV data
auto [X, y] = Dataset::load_csv("data.csv", {0, 1, 2}, {3});

// Preprocess data
Dataset::standardize(X);  // Standardize features
Matrix y_encoded = Dataset::one_hot_encode(y, num_classes);

// Generate synthetic data
auto [X_reg, y_reg] = Dataset::make_regression(1000, 10);  // 1000 samples, 10 features
auto [X_clf, y_clf] = Dataset::make_classification(1000, 10, 2);  // Binary classification
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Eigen library for advanced linear algebra operations
- Google Test for unit testing framework
- OpenMP for parallel processing support 