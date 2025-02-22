// Copyright 2024 Nesterov Alexander
#include "seq/opolin_d_sum_by_columns/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

using namespace std::chrono_literals;

bool opolin_d_sum_by_columns_seq::SumColumnsMatrixSequential::PreProcessingImpl() {
  // init data
  auto *ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_matrix_.assign(ptr, ptr + (rows_ * cols_));
  output_.resize(cols_, 0.0);
  return true;
}

bool opolin_d_sum_by_columns_seq::SumColumnsMatrixSequential::ValidationImpl() {
  // check input and output
  if (task_data->inputs_count.empty() || task_data->inputs.empty()) {
    return false;
  }
  if (task_data->outputs_count.empty() || task_data->inputs_count[1] != task_data->outputs_count[0] ||
      task_data->outputs.empty()) {
    return false;
  }
  rows_ = task_data->inputs_count[0];
  cols_ = task_data->inputs_count[1];
  return (rows_ > 0 && cols_ > 0);
}

bool opolin_d_sum_by_columns_seq::SumColumnsMatrixSequential::RunImpl() {
  // simple iteration method
  for (size_t col = 0; col < cols_; ++col) {
    for (size_t row = 0; row < rows_; ++row) {
      output_[col] += input_matrix_[(row * cols_) + col];
    }
  }
  return true;
}

bool opolin_d_sum_by_columns_seq::SumColumnsMatrixSequential::PostProcessingImpl() {
  auto *out = reinterpret_cast<int *>(task_data->outputs[0]);
  std::ranges::copy(output_, out);
  return true;
}