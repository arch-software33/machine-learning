#pragma once

#include "matrix.hpp"
#include <vector>
#include <memory>
#include <functional>

namespace mlcpp {

// Activation functions
enum class Activation {
    ReLU,
    Sigmoid,
    Tanh
};

// Layer interface
class Layer {
public:
    virtual ~Layer() = default;
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& gradient) = 0;
    virtual void update_parameters(double learning_rate) = 0;
};

// Dense (fully connected) layer
class DenseLayer : public Layer {
public:
    DenseLayer(size_t input_size, size_t output_size, Activation activation);
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_parameters(double learning_rate) override;
    
private:
    Matrix weights_;
    Matrix bias_;
    Matrix input_;
    Matrix output_before_activation_;
    Activation activation_;
    
    static Matrix apply_activation(const Matrix& x, Activation activation);
    static Matrix apply_activation_derivative(const Matrix& x, Activation activation);
};

// Neural Network class
class NeuralNetwork {
public:
    NeuralNetwork(double learning_rate = 0.01);
    
    // Network construction
    void add_layer(size_t input_size, size_t output_size, Activation activation);
    
    // Training methods
    void fit(const Matrix& X, const Matrix& y, size_t epochs, size_t batch_size = 32);
    Matrix predict(const Matrix& X) const;
    
    // Model evaluation
    double loss(const Matrix& y_true, const Matrix& y_pred) const;
    double accuracy(const Matrix& X, const Matrix& y) const;
    
private:
    std::vector<std::unique_ptr<Layer>> layers_;
    double learning_rate_;
    
    // Helper methods
    Matrix forward_propagation(const Matrix& X) const;
    void backward_propagation(const Matrix& X, const Matrix& y);
    void update_parameters();
    
    // Mini-batch processing
    std::vector<std::pair<Matrix, Matrix>> create_mini_batches(
        const Matrix& X, const Matrix& y, size_t batch_size) const;
};

} // namespace mlcpp 