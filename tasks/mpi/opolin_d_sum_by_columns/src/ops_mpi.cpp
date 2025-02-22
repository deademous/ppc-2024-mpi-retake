// Copyright 2024 Nesterov Alexander
#include "mpi/opolin_d_sum_by_columns/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

bool opolin_d_sum_by_columns_mpi::SumColumnsMatrixMPI::PreProcessingImpl() {
  // init data
  if (world_.rank() == 0) {
    auto *ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    input_matrix_.assign(ptr, ptr + (rows_ * cols_));
    output_.resize(cols_, 0);
  }
  return true;
}

bool opolin_d_sum_by_columns_mpi::SumColumnsMatrixMPI::ValidationImpl() {
  // check input and output
  if (world_.rank() == 0) {
    if (task_data->inputs_count.empty() || task_data->inputs.empty()) {
      return false;
    }
    if (task_data->outputs_count.empty() || task_data->inputs_count[1] != task_data->outputs_count[0] ||
        task_data->outputs.empty()) {
      return false;
    }
    rows_ = task_data->inputs_count[0];
    cols_ = task_data->inputs_count[1];
    if (rows_ <= 0 || cols_ <= 0) {
      return false;
    }
  }
  return true;
}

bool opolin_d_sum_by_columns_mpi::SumColumnsMatrixMPI::RunImpl() {
  broadcast(world_, rows_, 0);
  broadcast(world_, cols_, 0);
  auto proc_count = static_cast<size_t>(world_.size());
  auto proc_rank = static_cast<size_t>(world_.rank());
  auto remainder = rows_ % proc_count;
  size_t local_rows = rows_ / proc_count;
  if (proc_rank < (rows_ % proc_count)) {
    local_rows++;
  }
  std::vector<int> local_matrix(local_rows * cols_);
  std::vector<int> send_counts(world_.size(), 0);
  std::vector<int> displs(world_.size(), 0);
  std::vector<int> gathered_sums;

  if (world_.rank() == 0) {
    size_t offset = 0;
    for (int i = 0; i < world_.size(); ++i) {
      size_t rows_for_proc = rows_ / proc_count;
      if (remainder != 0 && i < static_cast<int>(remainder)) {
        rows_for_proc++;
      }
      send_counts[i] = static_cast<int>(rows_for_proc * cols_);
      displs[i] = static_cast<int>(offset);
      offset += rows_for_proc * cols_;
    }
  }
  if (world_.rank() == 0) {
    boost::mpi::scatterv(world_, input_matrix_.data(), send_counts, displs, local_matrix.data(),
                         static_cast<int>(local_rows * cols_), 0);
  } else {
    boost::mpi::scatterv(world_, local_matrix.data(), static_cast<int>(local_rows * cols_), 0);
  }

  std::vector<int> local_sum(cols_, 0);
  for (size_t row = 0; row < local_rows; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      local_sum[col] += local_matrix[(row * cols_) + col];
    }
  }
  if (world_.rank() == 0) {
    gathered_sums.resize(world_.size() * cols_);
  }
  boost::mpi::gather(world_, local_sum.data(), static_cast<int>(cols_), gathered_sums.data(), 0);
  if (world_.rank() == 0) {
    output_.assign(cols_, 0);
    for (int proc = 0; proc < world_.size(); ++proc) {
      for (size_t col = 0; col < cols_; ++col) {
        output_[col] += gathered_sums[(proc * cols_) + col];
      }
    }
  }
  return true;
}

bool opolin_d_sum_by_columns_mpi::SumColumnsMatrixMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto *out = reinterpret_cast<int *>(task_data->outputs[0]);
    std::ranges::copy(output_, out);
  }
  return true;
}