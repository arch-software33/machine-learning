#pragma once

#include "matrix.hpp"
#include <memory>

namespace mlcpp {

class LinearRegression {
public:
    LinearRegression(double learning_rate = 0.01, size_t max_iterations = 1000);
    
    // Training methods
    void fit(const Matrix& X, const Matrix& y);
    Matrix predict(const Matrix& X) const;
    
    // Model evaluation
    double score(const Matrix& X, const Matrix& y) const;
    double mse(const Matrix& X, const Matrix& y) const;
    double r2_score(const Matrix& X, const Matrix& y) const;
    
    // Getters
    const Matrix& get_weights() const { return weights_; }
    double get_bias() const { return bias_; }
    
private:
    Matrix weights_;
    double bias_;
    double learning_rate_;
    size_t max_iterations_;
    
    // Helper methods
    void initialize_parameters(size_t n_features);
    void gradient_descent_step(const Matrix& X, const Matrix& y);
};

} // namespace mlcpp 