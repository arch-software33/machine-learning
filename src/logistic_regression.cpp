#include "mlcpp/logistic_regression.hpp"
#include <cmath>

namespace mlcpp {

LogisticRegression::LogisticRegression(double learning_rate, size_t max_iterations)
    : bias_(0.0)
    , learning_rate_(learning_rate)
    , max_iterations_(max_iterations) {}

void LogisticRegression::initialize_parameters(size_t n_features) {
    weights_ = Matrix(n_features, 1);
    bias_ = 0.0;
}

Matrix LogisticRegression::sigmoid(const Matrix& z) {
    Matrix result(z.rows(), z.cols());
    
    // Apply sigmoid function with SIMD optimization
    for (size_t i = 0; i < z.rows(); ++i) {
        for (size_t j = 0; j < z.cols(); ++j) {
            result.at(i, j) = 1.0 / (1.0 + std::exp(-z.at(i, j)));
        }
    }
    
    return result;
}

void LogisticRegression::fit(const Matrix& X, const Matrix& y) {
    if (X.rows() != y.rows()) {
        throw std::invalid_argument("Number of samples in X and y must match");
    }
    
    initialize_parameters(X.cols());
    
    for (size_t iteration = 0; iteration < max_iterations_; ++iteration) {
        gradient_descent_step(X, y);
    }
}

void LogisticRegression::gradient_descent_step(const Matrix& X, const Matrix& y) {
    const size_t m = X.rows();
    
    // Forward propagation
    Matrix predictions = predict_proba(X);
    
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

Matrix LogisticRegression::predict_proba(const Matrix& X) const {
    // Compute X * weights + bias using SIMD operations
    Matrix z = X * weights_;
    
    // Add bias to all predictions
    for (size_t i = 0; i < z.rows(); ++i) {
        z.at(i, 0) += bias_;
    }
    
    return sigmoid(z);
}

Matrix LogisticRegression::predict(const Matrix& X) const {
    Matrix probabilities = predict_proba(X);
    Matrix predictions(probabilities.rows(), 1);
    
    // Convert probabilities to binary predictions
    for (size_t i = 0; i < probabilities.rows(); ++i) {
        predictions.at(i, 0) = probabilities.at(i, 0) >= 0.5 ? 1.0 : 0.0;
    }
    
    return predictions;
}

double LogisticRegression::accuracy_score(const Matrix& X, const Matrix& y) const {
    Matrix predictions = predict(X);
    size_t correct = 0;
    
    for (size_t i = 0; i < y.rows(); ++i) {
        if (std::abs(predictions.at(i, 0) - y.at(i, 0)) < 1e-10) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / y.rows();
}

double LogisticRegression::cross_entropy_loss(const Matrix& X, const Matrix& y) const {
    Matrix probabilities = predict_proba(X);
    double loss = 0.0;
    
    for (size_t i = 0; i < y.rows(); ++i) {
        double y_true = y.at(i, 0);
        double y_pred = probabilities.at(i, 0);
        
        // Avoid log(0)
        y_pred = std::max(std::min(y_pred, 1.0 - 1e-10), 1e-10);
        
        loss += y_true * std::log(y_pred) + (1.0 - y_true) * std::log(1.0 - y_pred);
    }
    
    return -loss / y.rows();
}

} // namespace mlcpp 