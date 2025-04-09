#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <MLFlow/Supervised/LinearRegression.h>
#include <MLFlow/Utils/io.h>
#include <omp.h>
#include <chrono>
using namespace MLflow::io;
using namespace MLflow::Supervised;

int main() {
    
    try {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<double>> X_data, y_data;
        std::ifstream file("diabetes_dataset.csv");
        std::string line;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string token;
            std::vector<double> row;
            while (std::getline(ss, token, ',')) {
                row.push_back(std::stod(token));
            }

            X_data.emplace_back(row.begin(), row.begin() + 10); // 10 features
            y_data.push_back({row[10]}); // 1 target
        }
        omp_set_num_threads(omp_get_max_threads());
        LinearRegression<> model(X_data, y_data);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> train_time = end - start;
        std::cout << "Training Time: " << train_time.count() << " ms\n";
        auto predictions = model.predict(X_data);

        double mse = 0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            double diff = predictions[i][0] - y_data[i][0];
            mse += diff * diff;
        }
        mse /= predictions.size();

        std::cout << "Linear Regression Test\n";
        std::cout << "Training MSE: " << mse << "\n";
        for (int i = 0; i < 5; ++i) {
            std::cout << "  Predicted: " << predictions[i][0]
                      << ", Actual: " << y_data[i][0] << "\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
