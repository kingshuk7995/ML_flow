#pragma once

#include <Eigen/Dense>
#include <omp.h>
#include <vector>
#include <stdexcept>
#include <utility>

namespace MLflow {
    namespace Supervised {

        template<typename X_type = double, typename y_type = double>
        class LinearRegression {
            using Matrix = Eigen::Matrix<X_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
            using Vector = Eigen::Matrix<X_type, Eigen::Dynamic, 1>;

            Matrix Weights;  // (n_features × n_outputs)
            Vector Bias;     // (n_outputs)

            void train(const Matrix& X_train, const Matrix& y_train) {
                int N = X_train.rows();
                Matrix X_aug(N, X_train.cols() + 1);
                X_aug << X_train, Matrix::Ones(N, 1); // Add bias

                Matrix W_full = (X_aug.transpose() * X_aug).ldlt().solve(X_aug.transpose() * y_train);
                Weights = W_full.topRows(X_train.cols());
                Bias = W_full.bottomRows(1);
            }

            // Copy-based conversion
            Matrix to_eigen_matrix(const std::vector<std::vector<X_type>>& vec) const {
                int rows = vec.size();
                int cols = vec[0].size();
                Matrix mat(rows, cols);

                #pragma omp parallel for
                for (int i = 0; i < rows; ++i)
                    for (int j = 0; j < cols; ++j)
                        mat(i, j) = vec[i][j];

                return mat;
            }

            // Move-based conversion (rvalue overload)
            Matrix to_eigen_matrix(std::vector<std::vector<X_type>>&& vec) const {
                int rows = vec.size();
                int cols = vec[0].size();
                Matrix mat(rows, cols);

                #pragma omp parallel for
                for (int i = 0; i < rows; ++i) {
                    auto&& row = std::move(vec[i]);
                    for (int j = 0; j < cols; ++j)
                        mat(i, j) = std::move(row[j]);
                }

                return mat;
            }

        public:
            // Constructor (copy)
            LinearRegression(const std::vector<std::vector<X_type>>& X_data,
                             const std::vector<std::vector<y_type>>& y_data) {
                Matrix X_mat = to_eigen_matrix(X_data);
                Matrix y_mat = to_eigen_matrix(y_data);
                train(X_mat, y_mat);
            }

            // Constructor (move)
            LinearRegression(std::vector<std::vector<X_type>>&& X_data,
                             std::vector<std::vector<y_type>>&& y_data) {
                Matrix X_mat = to_eigen_matrix(std::move(X_data));
                Matrix y_mat = to_eigen_matrix(std::move(y_data));
                train(X_mat, y_mat);
            }

            // Constructor (Eigen)
            LinearRegression(const Matrix& X_mat, const Matrix& y_mat) {
                train(X_mat, y_mat);
            }

            // Predict one sample
            std::vector<y_type> predict(const std::vector<X_type>& X_test) const {
                if (X_test.size() != Weights.rows()) {
                    throw std::invalid_argument("Input size does not match number of features.");
                }

                Eigen::Map<const Vector> x_vec(X_test.data(), X_test.size());
                Vector result = Weights.transpose() * x_vec + Bias;
                return std::vector<y_type>(result.data(), result.data() + result.size());
            }

            // Predict a batch
            std::vector<std::vector<y_type>> predict(const std::vector<std::vector<X_type>>& X_test_batch) const {
                int N = X_test_batch.size();
                int F = Weights.rows();
                int M = Weights.cols();

                std::vector<std::vector<y_type>> predictions(N, std::vector<y_type>(M));

                #pragma omp parallel for
                for (int i = 0; i < N; ++i) {
                    if (X_test_batch[i].size() != F)
                        throw std::invalid_argument("Inconsistent input feature size at row " + std::to_string(i));

                    Eigen::Map<const Vector> x_vec(X_test_batch[i].data(), F);
                    Vector result = Weights.transpose() * x_vec + Bias;
                    for (int j = 0; j < M; ++j)
                        predictions[i][j] = result(j);
                }

                return predictions;
            }

            ~LinearRegression() = default;
        };

    }
}