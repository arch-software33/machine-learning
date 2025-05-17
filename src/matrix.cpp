#include "mlcpp/matrix.hpp"
#include <random>
#include <algorithm>
#include <cstring>

namespace mlcpp {

Matrix::Matrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}

Matrix::Matrix(const Matrix& other)
    : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {}

Matrix::Matrix(Matrix&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), data_(std::move(other.data_)) {
    other.rows_ = 0;
    other.cols_ = 0;
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = other.data_;
    }
    return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = std::move(other.data_);
        other.rows_ = 0;
        other.cols_ = 0;
    }
    return *this;
}

void Matrix::check_dimensions(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }
}

size_t Matrix::index(size_t row, size_t col) const {
    return row * cols_ + col;
}

double& Matrix::at(size_t row, size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data_[index(row, col)];
}

const double& Matrix::at(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data_[index(row, col)];
}

// SIMD-optimized addition
void Matrix::add_simd(const Matrix& other) {
    check_dimensions(other);
    
    size_t aligned_size = (data_.size() / 4) * 4;
    size_t i = 0;
    
    // Process 4 doubles at a time using AVX
    for (; i < aligned_size; i += 4) {
        __m256d a = _mm256_loadu_pd(&data_[i]);
        __m256d b = _mm256_loadu_pd(&other.data_[i]);
        __m256d result = _mm256_add_pd(a, b);
        _mm256_storeu_pd(&data_[i], result);
    }
    
    // Handle remaining elements
    for (; i < data_.size(); ++i) {
        data_[i] += other.data_[i];
    }
}

// SIMD-optimized multiplication (matrix-vector multiplication)
void Matrix::multiply_simd(const Matrix& other) {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Invalid dimensions for matrix multiplication");
    }
    
    Matrix result(rows_, other.cols_);
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < other.cols_; ++j) {
            double sum = 0.0;
            size_t k = 0;
            
            // Process 4 elements at a time using AVX
            __m256d sum_vec = _mm256_setzero_pd();
            for (; k + 3 < cols_; k += 4) {
                __m256d a = _mm256_loadu_pd(&data_[i * cols_ + k]);
                __m256d b = _mm256_set_pd(
                    other.data_[(k + 3) * other.cols_ + j],
                    other.data_[(k + 2) * other.cols_ + j],
                    other.data_[(k + 1) * other.cols_ + j],
                    other.data_[k * other.cols_ + j]
                );
                __m256d prod = _mm256_mul_pd(a, b);
                sum_vec = _mm256_add_pd(sum_vec, prod);
            }
            
            // Horizontal sum
            double temp[4];
            _mm256_storeu_pd(temp, sum_vec);
            sum = temp[0] + temp[1] + temp[2] + temp[3];
            
            // Handle remaining elements
            for (; k < cols_; ++k) {
                sum += data_[i * cols_ + k] * other.data_[k * other.cols_ + j];
            }
            
            result.data_[i * other.cols_ + j] = sum;
        }
    }
    
    *this = std::move(result);
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.at(j, i) = at(i, j);
        }
    }
    return result;
}

// Eigen interoperability
Eigen::MatrixXd Matrix::toEigen() const {
    Eigen::MatrixXd result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result(i, j) = at(i, j);
        }
    }
    return result;
}

Matrix Matrix::fromEigen(const Eigen::MatrixXd& eigen_matrix) {
    Matrix result(eigen_matrix.rows(), eigen_matrix.cols());
    for (size_t i = 0; i < result.rows_; ++i) {
        for (size_t j = 0; j < result.cols_; ++j) {
            result.at(i, j) = eigen_matrix(i, j);
        }
    }
    return result;
}

// Utility functions
Matrix zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols);
}

Matrix ones(size_t rows, size_t cols) {
    Matrix result(rows, cols);
    std::fill(result.data_.begin(), result.data_.end(), 1.0);
    return result;
}

Matrix random(size_t rows, size_t cols, double min, double max) {
    Matrix result(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min, max);
    
    for (auto& val : result.data_) {
        val = dist(gen);
    }
    return result;
}

Matrix identity(size_t size) {
    Matrix result(size, size);
    for (size_t i = 0; i < size; ++i) {
        result.at(i, i) = 1.0;
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(*this);
    for (auto& val : result.data_) {
        val *= scalar;
    }
    return result;
}

Matrix& Matrix::operator*=(double scalar) {
    for (auto& val : data_) {
        val *= scalar;
    }
    return *this;
}

Matrix Matrix::operator+(const Matrix& other) const {
    check_dimensions(other);
    Matrix result(*this);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] += other.data_[i];
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    check_dimensions(other);
    Matrix result(*this);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] -= other.data_[i];
    }
    return result;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    check_dimensions(other);
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    check_dimensions(other);
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] -= other.data_[i];
    }
    return *this;
}

Matrix Matrix::hadamard(const Matrix& other) const {
    check_dimensions(other);
    Matrix result(*this);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] *= other.data_[i];
    }
    return result;
}

Matrix Matrix::apply(double (*func)(double)) const {
    Matrix result(*this);
    for (auto& val : result.data_) {
        val = func(val);
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Invalid dimensions for matrix multiplication");
    }
    
    Matrix result(rows_, other.cols_);
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < other.cols_; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < cols_; ++k) {
                sum += at(i, k) * other.at(k, j);
            }
            result.at(i, j) = sum;
        }
    }
    
    return result;
}

} // namespace mlcpp 