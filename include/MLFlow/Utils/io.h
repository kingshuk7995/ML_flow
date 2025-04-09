#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <tuple>
#include <stdexcept>
#include <utility>

namespace MLflow {
    namespace io {      //io related works
        using std::string;
        using std::vector;
        using std::ifstream;
        using std::stringstream;

        // Utility to split a line based on delimiter
        inline vector<string> split_line(const string& line, char delimiter) {
            vector<string> tokens;
            string token;
            std::istringstream stream(line);
            while (std::getline(stream, token, delimiter)) {
                tokens.push_back(token);
            }
            return tokens;
        }

        // Helper: Convert string to a specific type
        template<typename T>
        T convert(const string& s) {
            std::istringstream iss(s);
            T value;
            if (!(iss >> value)) {
                throw std::runtime_error("Conversion failed for value: " + s);
            }
            return value;
        }

        // Specialization for string (no conversion needed)
        template<>
        inline string convert<string>(const string& s) {
            return s;
        }

        // Helper to read a line and convert to tuple
        template<typename... Types, size_t... Is>
        std::tuple<Types...> parse_tuple_impl(const vector<string>& tokens, std::index_sequence<Is...>) {
            if (tokens.size() != sizeof...(Types)) {
                throw std::runtime_error("Column count mismatch while parsing line.");
            }
            return std::make_tuple(convert<Types>(tokens[Is])...);
        }

        template<typename... Types>
        std::tuple<Types...> parse_tuple(const vector<string>& tokens) {
            return parse_tuple_impl<Types...>(tokens, std::index_sequence_for<Types...>{});
        }

        // Main class template
        template<typename... Types>
        class Separated_files {
            vector<std::tuple<Types...>> data_;
            char delimiter_;

        public:
            Separated_files(const string& filename, char delimiter = ',') : delimiter_(delimiter) {
                std::ifstream file(filename);
                if (!file.is_open()) {
                    throw std::runtime_error("Failed to open file: " + filename);
                }

                string line;
                while (std::getline(file, line)) {
                    auto tokens = split_line(line, delimiter_);
                    auto tuple = parse_tuple<Types...>(tokens);
                    data_.push_back(tuple);
                }
            }

            const vector<std::tuple<Types...>>& data() const {
                return data_;
            }

            size_t size() const {
                return data_.size();
            }

            const std::tuple<Types...>& operator[](size_t index) const {
                return data_.at(index);
            }
        };
    }
}
