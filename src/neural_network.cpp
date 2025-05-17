#include "mlcpp/neural_network.hpp"
#include <random>
#include <algorithm>
#include <cmath>

namespace mlcpp {

// DenseLayer implementation
DenseLayer::DenseLayer(size_t input_size, size_t output_size, Activation activation)
    : activation_(activation) {
    // Xavier/Glorot initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    double limit = std::sqrt(6.0 / (input_size + output_size));
    std::uniform_real_distribution<double> dist(-limit, limit);
    
    weights_ = Matrix(input_size, output_size);
    bias_ = Matrix(1, output_size);
    
    // Initialize weights and bias
    for (size_t i = 0; i < input_size; ++i) {
        for (size_t j = 0; j < output_size; ++j) {
            weights_.at(i, j) = dist(gen);
        }
    }
    
    for (size_t j = 0; j < output_size; ++j) {
        bias_.at(0, j) = 0.0;
    }
}

Matrix DenseLayer::apply_activation(const Matrix& x, Activation activation) {
    Matrix result(x.rows(), x.cols());
    
    switch (activation) {
        case Activation::ReLU:
            for (size_t i = 0; i < x.rows(); ++i) {
                for (size_t j = 0; j < x.cols(); ++j) {
                    result.at(i, j) = std::max(0.0, x.at(i, j));
                }
            }
            break;
            
        case Activation::Sigmoid:
            for (size_t i = 0; i < x.rows(); ++i) {
                for (size_t j = 0; j < x.cols(); ++j) {
                    result.at(i, j) = 1.0 / (1.0 + std::exp(-x.at(i, j)));
                }
            }
            break;
            
        case Activation::Tanh:
            for (size_t i = 0; i < x.rows(); ++i) {
                for (size_t j = 0; j < x.cols(); ++j) {
                    result.at(i, j) = std::tanh(x.at(i, j));
                }
            }
            break;
    }
    
    return result;
}

Matrix DenseLayer::apply_activation_derivative(const Matrix& x, Activation activation) {
    Matrix result(x.rows(), x.cols());
    
    switch (activation) {
        case Activation::ReLU:
            for (size_t i = 0; i < x.rows(); ++i) {
                for (size_t j = 0; j < x.cols(); ++j) {
                    result.at(i, j) = x.at(i, j) > 0 ? 1.0 : 0.0;
                }
            }
            break;
            
        case Activation::Sigmoid:
            for (size_t i = 0; i < x.rows(); ++i) {
                for (size_t j = 0; j < x.cols(); ++j) {
                    double sigmoid = 1.0 / (1.0 + std::exp(-x.at(i, j)));
                    result.at(i, j) = sigmoid * (1.0 - sigmoid);
                }
            }
            break;
            
        case Activation::Tanh:
            for (size_t i = 0; i < x.rows(); ++i) {
                for (size_t j = 0; j < x.cols(); ++j) {
                    double tanh_x = std::tanh(x.at(i, j));
                    result.at(i, j) = 1.0 - tanh_x * tanh_x;
                }
            }
            break;
    }
    
    return result;
}

Matrix DenseLayer::forward(const Matrix& input) {
    input_ = input;
    
    // Linear transformation with SIMD
    output_before_activation_ = input * weights_;
    for (size_t i = 0; i < output_before_activation_.rows(); ++i) {
        for (size_t j = 0; j < output_before_activation_.cols(); ++j) {
            output_before_activation_.at(i, j) += bias_.at(0, j);
        }
    }
    
    // Apply activation function
    return apply_activation(output_before_activation_, activation_);
}

Matrix DenseLayer::backward(const Matrix& gradient) {
    // Calculate activation gradient
    Matrix activation_gradient = apply_activation_derivative(output_before_activation_, activation_);
    Matrix delta = gradient.hadamard(activation_gradient);
    
    // Calculate gradients with SIMD
    Matrix weight_gradients = input_.transpose() * delta;
    Matrix input_gradients = delta * weights_.transpose();
    
    // Update gradients
    weights_ -= weight_gradients;
    
    // Calculate bias gradients
    for (size_t j = 0; j < bias_.cols(); ++j) {
        double bias_gradient = 0.0;
        for (size_t i = 0; i < delta.rows(); ++i) {
            bias_gradient += delta.at(i, j);
        }
        bias_.at(0, j) -= bias_gradient;
    }
    
    return input_gradients;
}

void DenseLayer::update_parameters(double learning_rate) {
    // Update is done in backward pass for simplicity
}

// Neural Network implementation
NeuralNetwork::NeuralNetwork(double learning_rate)
    : learning_rate_(learning_rate) {}

void NeuralNetwork::add_layer(size_t input_size, size_t output_size, Activation activation) {
    layers_.push_back(std::make_unique<DenseLayer>(input_size, output_size, activation));
}

Matrix NeuralNetwork::forward_propagation(const Matrix& X) const {
    Matrix current = X;
    for (const auto& layer : layers_) {
        current = layer->forward(current);
    }
    return current;
}

void NeuralNetwork::backward_propagation(const Matrix& X, const Matrix& y) {
    Matrix gradient = forward_propagation(X);
    gradient -= y;  // Assuming MSE loss for simplicity
    
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        gradient = (*it)->backward(gradient);
    }
}

void NeuralNetwork::update_parameters() {
    for (auto& layer : layers_) {
        layer->update_parameters(learning_rate_);
    }
}

std::vector<std::pair<Matrix, Matrix>> NeuralNetwork::create_mini_batches(
    const Matrix& X, const Matrix& y, size_t batch_size) const {
    std::vector<std::pair<Matrix, Matrix>> mini_batches;
    
    // Create indices and shuffle
    std::vector<size_t> indices(X.rows());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    
    // Create mini-batches
    for (size_t i = 0; i < X.rows(); i += batch_size) {
        size_t current_batch_size = std::min(batch_size, X.rows() - i);
        
        Matrix batch_X(current_batch_size, X.cols());
        Matrix batch_y(current_batch_size, y.cols());
        
        for (size_t j = 0; j < current_batch_size; ++j) {
            size_t idx = indices[i + j];
            for (size_t k = 0; k < X.cols(); ++k) {
                batch_X.at(j, k) = X.at(idx, k);
            }
            for (size_t k = 0; k < y.cols(); ++k) {
                batch_y.at(j, k) = y.at(idx, k);
            }
        }
        
        mini_batches.emplace_back(batch_X, batch_y);
    }
    
    return mini_batches;
}

void NeuralNetwork::fit(const Matrix& X, const Matrix& y, size_t epochs, size_t batch_size) {
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        auto mini_batches = create_mini_batches(X, y, batch_size);
        
        for (const auto& batch : mini_batches) {
            backward_propagation(batch.first, batch.second);
            update_parameters();
        }
    }
}

Matrix NeuralNetwork::predict(const Matrix& X) const {
    return forward_propagation(X);
}

double NeuralNetwork::loss(const Matrix& y_true, const Matrix& y_pred) const {
    double mse = 0.0;
    for (size_t i = 0; i < y_true.rows(); ++i) {
        for (size_t j = 0; j < y_true.cols(); ++j) {
            double diff = y_true.at(i, j) - y_pred.at(i, j);
            mse += diff * diff;
        }
    }
    return mse / (y_true.rows() * y_true.cols());
}

double NeuralNetwork::accuracy(const Matrix& X, const Matrix& y) const {
    Matrix predictions = predict(X);
    size_t correct = 0;
    
    for (size_t i = 0; i < y.rows(); ++i) {
        size_t pred_class = 0;
        size_t true_class = 0;
        double max_pred = predictions.at(i, 0);
        double max_true = y.at(i, 0);
        
        for (size_t j = 1; j < y.cols(); ++j) {
            if (predictions.at(i, j) > max_pred) {
                max_pred = predictions.at(i, j);
                pred_class = j;
            }
            if (y.at(i, j) > max_true) {
                max_true = y.at(i, j);
                true_class = j;
            }
        }
        
        if (pred_class == true_class) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / y.rows();
}

} // namespace mlcpp 