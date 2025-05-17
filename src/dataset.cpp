#include "mlcpp/dataset.hpp"
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mlcpp {

std::vector<std::string> Dataset::split_line(const std::string& line, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    
    return tokens;
}

std::pair<Matrix, Matrix> Dataset::load_csv(
    const std::string& filename,
    const std::vector<size_t>& feature_cols,
    const std::vector<size_t>& target_cols,
    bool has_header) {
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    std::string line;
    std::vector<std::vector<double>> features;
    std::vector<std::vector<double>> targets;
    
    // Skip header if present
    if (has_header) {
        std::getline(file, line);
    }
    
    // Read data
    while (std::getline(file, line)) {
        auto tokens = split_line(line);
        
        if (tokens.size() <= std::max(
            *std::max_element(feature_cols.begin(), feature_cols.end()),
            *std::max_element(target_cols.begin(), target_cols.end()))) {
            continue;  // Skip invalid lines
        }
        
        std::vector<double> feature_row;
        std::vector<double> target_row;
        
        for (size_t col : feature_cols) {
            feature_row.push_back(std::stod(tokens[col]));
        }
        
        for (size_t col : target_cols) {
            target_row.push_back(std::stod(tokens[col]));
        }
        
        features.push_back(feature_row);
        targets.push_back(target_row);
    }
    
    // Convert to matrices
    Matrix X(features.size(), feature_cols.size());
    Matrix y(targets.size(), target_cols.size());
    
    for (size_t i = 0; i < features.size(); ++i) {
        for (size_t j = 0; j < feature_cols.size(); ++j) {
            X.at(i, j) = features[i][j];
        }
        for (size_t j = 0; j < target_cols.size(); ++j) {
            y.at(i, j) = targets[i][j];
        }
    }
    
    return {X, y};
}

void Dataset::normalize(Matrix& X, std::vector<double>& means, std::vector<double>& stds) {
    means.resize(X.cols());
    stds.resize(X.cols());
    
    // Calculate means and standard deviations
    for (size_t j = 0; j < X.cols(); ++j) {
        double sum = 0.0;
        double sum_sq = 0.0;
        
        for (size_t i = 0; i < X.rows(); ++i) {
            sum += X.at(i, j);
            sum_sq += X.at(i, j) * X.at(i, j);
        }
        
        means[j] = sum / X.rows();
        stds[j] = std::sqrt(sum_sq / X.rows() - means[j] * means[j]);
        
        // Avoid division by zero
        if (std::abs(stds[j]) < 1e-10) {
            stds[j] = 1.0;
        }
    }
    
    // Normalize the data
    for (size_t i = 0; i < X.rows(); ++i) {
        for (size_t j = 0; j < X.cols(); ++j) {
            X.at(i, j) = (X.at(i, j) - means[j]) / stds[j];
        }
    }
}

void Dataset::standardize(Matrix& X) {
    std::vector<double> means, stds;
    normalize(X, means, stds);
}

Matrix Dataset::one_hot_encode(const Matrix& y, size_t num_classes) {
    Matrix encoded(y.rows(), num_classes);
    
    for (size_t i = 0; i < y.rows(); ++i) {
        size_t class_idx = static_cast<size_t>(y.at(i, 0));
        if (class_idx >= num_classes) {
            throw std::runtime_error("Class index out of range in one-hot encoding");
        }
        encoded.at(i, class_idx) = 1.0;
    }
    
    return encoded;
}

void Dataset::shuffle_data(Matrix& X, Matrix& y) {
    std::vector<size_t> indices(X.rows());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    
    Matrix X_shuffled(X.rows(), X.cols());
    Matrix y_shuffled(y.rows(), y.cols());
    
    for (size_t i = 0; i < indices.size(); ++i) {
        for (size_t j = 0; j < X.cols(); ++j) {
            X_shuffled.at(i, j) = X.at(indices[i], j);
        }
        for (size_t j = 0; j < y.cols(); ++j) {
            y_shuffled.at(i, j) = y.at(indices[i], j);
        }
    }
    
    X = std::move(X_shuffled);
    y = std::move(y_shuffled);
}

std::tuple<Matrix, Matrix, Matrix, Matrix> Dataset::train_test_split(
    const Matrix& X, const Matrix& y, double test_size, bool shuffle) {
    
    if (X.rows() != y.rows()) {
        throw std::invalid_argument("X and y must have the same number of samples");
    }
    
    Matrix X_copy = X;
    Matrix y_copy = y;
    
    if (shuffle) {
        shuffle_data(X_copy, y_copy);
    }
    
    size_t test_samples = static_cast<size_t>(X.rows() * test_size);
    size_t train_samples = X.rows() - test_samples;
    
    Matrix X_train(train_samples, X.cols());
    Matrix X_test(test_samples, X.cols());
    Matrix y_train(train_samples, y.cols());
    Matrix y_test(test_samples, y.cols());
    
    // Split the data
    for (size_t i = 0; i < train_samples; ++i) {
        for (size_t j = 0; j < X.cols(); ++j) {
            X_train.at(i, j) = X_copy.at(i, j);
        }
        for (size_t j = 0; j < y.cols(); ++j) {
            y_train.at(i, j) = y_copy.at(i, j);
        }
    }
    
    for (size_t i = 0; i < test_samples; ++i) {
        for (size_t j = 0; j < X.cols(); ++j) {
            X_test.at(i, j) = X_copy.at(train_samples + i, j);
        }
        for (size_t j = 0; j < y.cols(); ++j) {
            y_test.at(i, j) = y_copy.at(train_samples + i, j);
        }
    }
    
    return {X_train, X_test, y_train, y_test};
}

std::pair<Matrix, Matrix> Dataset::make_regression(
    size_t n_samples, size_t n_features, double noise) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> normal(0.0, 1.0);
    
    // Generate random features
    Matrix X(n_samples, n_features);
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            X.at(i, j) = normal(gen);
        }
    }
    
    // Generate random weights
    std::vector<double> true_weights(n_features);
    for (size_t j = 0; j < n_features; ++j) {
        true_weights[j] = normal(gen);
    }
    
    // Generate target values
    Matrix y(n_samples, 1);
    for (size_t i = 0; i < n_samples; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < n_features; ++j) {
            sum += X.at(i, j) * true_weights[j];
        }
        y.at(i, 0) = sum + noise * normal(gen);
    }
    
    return {X, y};
}

std::pair<Matrix, Matrix> Dataset::make_classification(
    size_t n_samples, size_t n_features, size_t n_classes, double class_sep) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_int_distribution<size_t> class_dist(0, n_classes - 1);
    
    // Generate class centers
    std::vector<std::vector<double>> centers(n_classes, std::vector<double>(n_features));
    for (size_t i = 0; i < n_classes; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            centers[i][j] = class_sep * normal(gen);
        }
    }
    
    // Generate samples around centers
    Matrix X(n_samples, n_features);
    Matrix y(n_samples, 1);
    
    for (size_t i = 0; i < n_samples; ++i) {
        size_t class_idx = class_dist(gen);
        y.at(i, 0) = static_cast<double>(class_idx);
        
        for (size_t j = 0; j < n_features; ++j) {
            X.at(i, j) = centers[class_idx][j] + 0.1 * normal(gen);
        }
    }
    
    return {X, y};
}

} // namespace mlcpp 