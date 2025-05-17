#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include <immintrin.h>
#include <Eigen/Dense>

namespace mlcpp {

class Matrix {
public:
    // Constructors
    Matrix(size_t rows, size_t cols);
    Matrix(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    
    // Assignment operators
    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other) noexcept;
    
    // Basic operations
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    
    // Element-wise operations
    Matrix hadamard(const Matrix& other) const;
    Matrix apply(double (*func)(double)) const;
    
    // Matrix operations
    Matrix transpose() const;
    Matrix inverse() const;
    double determinant() const;
    
    // Getters and setters
    double& at(size_t row, size_t col);
    const double& at(size_t row, size_t col) const;
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    
    // SIMD optimized operations
    void add_simd(const Matrix& other);
    void subtract_simd(const Matrix& other);
    void multiply_simd(const Matrix& other);
    
    // Eigen interoperability
    Eigen::MatrixXd toEigen() const;
    static Matrix fromEigen(const Eigen::MatrixXd& eigen_matrix);

private:
    size_t rows_;
    size_t cols_;
    std::vector<double> data_;
    
    // Helper functions
    void check_dimensions(const Matrix& other) const;
    size_t index(size_t row, size_t col) const;
};

// Vector alias
using Vector = Matrix;

// Utility functions
Matrix zeros(size_t rows, size_t cols);
Matrix ones(size_t rows, size_t cols);
Matrix random(size_t rows, size_t cols, double min = 0.0, double max = 1.0);
Matrix identity(size_t size);

} // namespace mlcpp 