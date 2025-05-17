#pragma once

#include "matrix.hpp"
#include <string>
#include <vector>
#include <utility>

namespace mlcpp {

class Dataset {
public:
    // Data loading
    static std::pair<Matrix, Matrix> load_csv(const std::string& filename,
                                            const std::vector<size_t>& feature_cols,
                                            const std::vector<size_t>& target_cols,
                                            bool has_header = true);
    
    // Data preprocessing
    static void normalize(Matrix& X, std::vector<double>& means, std::vector<double>& stds);
    static void standardize(Matrix& X);
    static Matrix one_hot_encode(const Matrix& y, size_t num_classes);
    
    // Train-test split
    static std::tuple<Matrix, Matrix, Matrix, Matrix> train_test_split(
        const Matrix& X, const Matrix& y, double test_size = 0.2, bool shuffle = true);
    
    // Data generation
    static std::pair<Matrix, Matrix> make_regression(
        size_t n_samples = 100,
        size_t n_features = 1,
        double noise = 0.1);
        
    static std::pair<Matrix, Matrix> make_classification(
        size_t n_samples = 100,
        size_t n_features = 2,
        size_t n_classes = 2,
        double class_sep = 1.0);

private:
    static void shuffle_data(Matrix& X, Matrix& y);
    static std::vector<std::string> split_line(const std::string& line, char delimiter = ',');
};

} // namespace mlcpp 