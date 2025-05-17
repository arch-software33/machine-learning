#include <gtest/gtest.h>
#include "mlcpp/matrix.hpp"

using namespace mlcpp;

class MatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test matrices
        mat1 = Matrix(2, 2);
        mat2 = Matrix(2, 2);
        
        // Initialize mat1
        mat1.at(0, 0) = 1.0;
        mat1.at(0, 1) = 2.0;
        mat1.at(1, 0) = 3.0;
        mat1.at(1, 1) = 4.0;
        
        // Initialize mat2
        mat2.at(0, 0) = 5.0;
        mat2.at(0, 1) = 6.0;
        mat2.at(1, 0) = 7.0;
        mat2.at(1, 1) = 8.0;
    }
    
    Matrix mat1;
    Matrix mat2;
};

TEST_F(MatrixTest, Construction) {
    Matrix mat(3, 4);
    EXPECT_EQ(mat.rows(), 3);
    EXPECT_EQ(mat.cols(), 4);
    
    // Check initialization to zero
    for (size_t i = 0; i < mat.rows(); ++i) {
        for (size_t j = 0; j < mat.cols(); ++j) {
            EXPECT_DOUBLE_EQ(mat.at(i, j), 0.0);
        }
    }
}

TEST_F(MatrixTest, CopyConstruction) {
    Matrix copy(mat1);
    EXPECT_EQ(copy.rows(), mat1.rows());
    EXPECT_EQ(copy.cols(), mat1.cols());
    
    for (size_t i = 0; i < copy.rows(); ++i) {
        for (size_t j = 0; j < copy.cols(); ++j) {
            EXPECT_DOUBLE_EQ(copy.at(i, j), mat1.at(i, j));
        }
    }
}

TEST_F(MatrixTest, Addition) {
    Matrix sum = mat1 + mat2;
    EXPECT_EQ(sum.rows(), 2);
    EXPECT_EQ(sum.cols(), 2);
    
    EXPECT_DOUBLE_EQ(sum.at(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(sum.at(0, 1), 8.0);
    EXPECT_DOUBLE_EQ(sum.at(1, 0), 10.0);
    EXPECT_DOUBLE_EQ(sum.at(1, 1), 12.0);
}

TEST_F(MatrixTest, Multiplication) {
    Matrix prod = mat1 * mat2;
    EXPECT_EQ(prod.rows(), 2);
    EXPECT_EQ(prod.cols(), 2);
    
    EXPECT_DOUBLE_EQ(prod.at(0, 0), 19.0);
    EXPECT_DOUBLE_EQ(prod.at(0, 1), 22.0);
    EXPECT_DOUBLE_EQ(prod.at(1, 0), 43.0);
    EXPECT_DOUBLE_EQ(prod.at(1, 1), 50.0);
}

TEST_F(MatrixTest, Transpose) {
    Matrix trans = mat1.transpose();
    EXPECT_EQ(trans.rows(), mat1.cols());
    EXPECT_EQ(trans.cols(), mat1.rows());
    
    for (size_t i = 0; i < mat1.rows(); ++i) {
        for (size_t j = 0; j < mat1.cols(); ++j) {
            EXPECT_DOUBLE_EQ(trans.at(j, i), mat1.at(i, j));
        }
    }
}

TEST_F(MatrixTest, HadamardProduct) {
    Matrix hadamard = mat1.hadamard(mat2);
    EXPECT_EQ(hadamard.rows(), 2);
    EXPECT_EQ(hadamard.cols(), 2);
    
    EXPECT_DOUBLE_EQ(hadamard.at(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(hadamard.at(0, 1), 12.0);
    EXPECT_DOUBLE_EQ(hadamard.at(1, 0), 21.0);
    EXPECT_DOUBLE_EQ(hadamard.at(1, 1), 32.0);
}

TEST_F(MatrixTest, EigenInterop) {
    Eigen::MatrixXd eigen_mat = mat1.toEigen();
    EXPECT_EQ(eigen_mat.rows(), mat1.rows());
    EXPECT_EQ(eigen_mat.cols(), mat1.cols());
    
    Matrix converted = Matrix::fromEigen(eigen_mat);
    EXPECT_EQ(converted.rows(), mat1.rows());
    EXPECT_EQ(converted.cols(), mat1.cols());
    
    for (size_t i = 0; i < mat1.rows(); ++i) {
        for (size_t j = 0; j < mat1.cols(); ++j) {
            EXPECT_DOUBLE_EQ(converted.at(i, j), mat1.at(i, j));
        }
    }
}

TEST_F(MatrixTest, SIMDOperations) {
    // Test SIMD-optimized addition
    Matrix sum(2, 2);
    mat1.add_simd(mat2);
    
    EXPECT_DOUBLE_EQ(mat1.at(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(mat1.at(0, 1), 8.0);
    EXPECT_DOUBLE_EQ(mat1.at(1, 0), 10.0);
    EXPECT_DOUBLE_EQ(mat1.at(1, 1), 12.0);
    
    // Test SIMD-optimized multiplication
    Matrix A(2, 3);
    Matrix B(3, 2);
    
    // Initialize test matrices
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            A.at(i, j) = i + j;
        }
    }
    
    for (size_t i = 0; i < B.rows(); ++i) {
        for (size_t j = 0; j < B.cols(); ++j) {
            B.at(i, j) = i - j;
        }
    }
    
    A.multiply_simd(B);
    
    // Verify results against manual calculation
    Matrix expected(2, 2);
    expected.at(0, 0) = 5.0;
    expected.at(0, 1) = 2.0;
    expected.at(1, 0) = 8.0;
    expected.at(1, 1) = 5.0;
    
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < B.cols(); ++j) {
            EXPECT_NEAR(A.at(i, j), expected.at(i, j), 1e-10);
        }
    }
}

TEST_F(MatrixTest, UtilityFunctions) {
    // Test zeros
    Matrix zero = zeros(2, 3);
    EXPECT_EQ(zero.rows(), 2);
    EXPECT_EQ(zero.cols(), 3);
    for (size_t i = 0; i < zero.rows(); ++i) {
        for (size_t j = 0; j < zero.cols(); ++j) {
            EXPECT_DOUBLE_EQ(zero.at(i, j), 0.0);
        }
    }
    
    // Test ones
    Matrix one = ones(2, 3);
    EXPECT_EQ(one.rows(), 2);
    EXPECT_EQ(one.cols(), 3);
    for (size_t i = 0; i < one.rows(); ++i) {
        for (size_t j = 0; j < one.cols(); ++j) {
            EXPECT_DOUBLE_EQ(one.at(i, j), 1.0);
        }
    }
    
    // Test identity
    Matrix eye = identity(3);
    EXPECT_EQ(eye.rows(), 3);
    EXPECT_EQ(eye.cols(), 3);
    for (size_t i = 0; i < eye.rows(); ++i) {
        for (size_t j = 0; j < eye.cols(); ++j) {
            EXPECT_DOUBLE_EQ(eye.at(i, j), i == j ? 1.0 : 0.0);
        }
    }
    
    // Test random
    Matrix rand = random(2, 3);
    EXPECT_EQ(rand.rows(), 2);
    EXPECT_EQ(rand.cols(), 3);
    bool has_different_values = false;
    double first_value = rand.at(0, 0);
    
    for (size_t i = 0; i < rand.rows(); ++i) {
        for (size_t j = 0; j < rand.cols(); ++j) {
            if (std::abs(rand.at(i, j) - first_value) > 1e-10) {
                has_different_values = true;
                break;
            }
        }
    }
    
    EXPECT_TRUE(has_different_values);
} 