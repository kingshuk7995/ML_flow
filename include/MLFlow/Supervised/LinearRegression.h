#pragma once
#include <Eigen/Dense>
#include <omp.h>
#include <vector>
#include <stdexcept>

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

        public:
            LinearRegression(const std::vector<std::vector<X_type>>& X_data,
                             const std::vector<std::vector<y_type>>& y_data) {
                int rows = X_data.size();
                int cols = X_data[0].size();
                int outputs = y_data[0].size();

                Matrix X_mat(rows, cols);
                Matrix y_mat(rows, outputs);

                #pragma omp parallel for
                for (int i = 0; i < rows; ++i) {
                    for (int j = 0; j < cols; ++j)
                        X_mat(i, j) = X_data[i][j];
                    for (int k = 0; k < outputs; ++k)
                        y_mat(i, k) = y_data[i][k];
                }

                train(X_mat, y_mat);
            }

            std::vector<y_type> predict(const std::vector<X_type>& X_test) const {
                if (X_test.size() != Weights.rows()) {
                    throw std::invalid_argument("Input size does not match number of features.");
                }

                Eigen::Map<const Eigen::Matrix<X_type, Eigen::Dynamic, 1>> x_vec(X_test.data(), X_test.size());
                Eigen::Matrix<X_type, Eigen::Dynamic, 1> result = Weights.transpose() * x_vec + Bias;

                return std::vector<y_type>(result.data(), result.data() + result.size());
            }

            // Batch Prediction
            std::vector<std::vector<y_type>> predict(const std::vector<std::vector<X_type>>& X_test_batch) const {
                int N = X_test_batch.size();
                int F = Weights.rows();
                int M = Weights.cols();

                std::vector<std::vector<y_type>> predictions(N, std::vector<y_type>(M));

                #pragma omp parallel for
                for (int i = 0; i < N; ++i) {
                    if (X_test_batch[i].size() != F)
                        throw std::invalid_argument("Inconsistent input feature size at row " + std::to_string(i));

                    Eigen::Map<const Eigen::Matrix<X_type, Eigen::Dynamic, 1>> x_vec(X_test_batch[i].data(), F);
                    Eigen::Matrix<X_type, Eigen::Dynamic, 1> result = Weights.transpose() * x_vec + Bias;

                    for (int j = 0; j < M; ++j)
                        predictions[i][j] = result(j);
                }

                return predictions;
            }

            ~LinearRegression() = default;
        };
    }
}
