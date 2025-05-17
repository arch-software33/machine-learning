#include "mlcpp/linear_regression.hpp"
#include <cmath>

namespace mlcpp {

LinearRegression::LinearRegression(double learning_rate, size_t max_iterations)
    : bias_(0.0)
    , learning_rate_(learning_rate)
    , max_iterations_(max_iterations) {}

void LinearRegression::initialize_parameters(size_t n_features) {
    weights_ = Matrix(n_features, 1);
    bias_ = 0.0;
}

void LinearRegression::fit(const Matrix& X, const Matrix& y) {
    if (X.rows() != y.rows()) {
        throw std::invalid_argument("Number of samples in X and y must match");
    }
    
    initialize_parameters(X.cols());
    
    for (size_t iteration = 0; iteration < max_iterations_; ++iteration) {
        gradient_descent_step(X, y);
    }
}

void LinearRegression::gradient_descent_step(const Matrix& X, const Matrix& y) {
    const size_t m = X.rows();
    
    // Calculate predictions using SIMD
    Matrix predictions = predict(X);
    
    // Calculate gradients
    Matrix error = predictions;
    error -= y;  // element-wise subtraction
    
    // Calculate gradients for weights using SIMD matrix operations
    Matrix X_transpose = X.transpose();
    Matrix weight_gradients = X_transpose * error;
    weight_gradients *= (1.0 / m);
    
    // Calculate gradient for bias
    double bias_gradient = 0.0;
    for (size_t i = 0; i < m; ++i) {
        bias_gradient += error.at(i, 0);
    }
    bias_gradient /= m;
    
    // Update parameters using SIMD
    weights_ -= weight_gradients * learning_rate_;
    bias_ -= learning_rate_ * bias_gradient;
}

Matrix LinearRegression::predict(const Matrix& X) const {
    // Compute X * weights + bias using SIMD operations
    Matrix predictions = X * weights_;
    
    // Add bias to all predictions
    for (size_t i = 0; i < predictions.rows(); ++i) {
        predictions.at(i, 0) += bias_;
    }
    
    return predictions;
}

double LinearRegression::mse(const Matrix& X, const Matrix& y) const {
    Matrix predictions = predict(X);
    Matrix error = predictions - y;
    
    double mse = 0.0;
    for (size_t i = 0; i < error.rows(); ++i) {
        mse += error.at(i, 0) * error.at(i, 0);
    }
    
    return mse / error.rows();
}

double LinearRegression::r2_score(const Matrix& X, const Matrix& y) const {
    Matrix predictions = predict(X);
    
    double y_mean = 0.0;
    for (size_t i = 0; i < y.rows(); ++i) {
        y_mean += y.at(i, 0);
    }
    y_mean /= y.rows();
    
    double ss_total = 0.0;
    double ss_residual = 0.0;
    
    for (size_t i = 0; i < y.rows(); ++i) {
        double y_true = y.at(i, 0);
        double y_pred = predictions.at(i, 0);
        
        ss_total += (y_true - y_mean) * (y_true - y_mean);
        ss_residual += (y_true - y_pred) * (y_true - y_pred);
    }
    
    return 1.0 - (ss_residual / ss_total);
}

double LinearRegression::score(const Matrix& X, const Matrix& y) const {
    return r2_score(X, y);
}

} // namespace mlcpp 