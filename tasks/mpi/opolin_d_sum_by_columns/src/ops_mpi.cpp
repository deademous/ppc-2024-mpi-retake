// Copyright 2024 Nesterov Alexander
#include "mpi/opolin_d_sum_by_columns/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(misc-include-cleaner)
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

bool opolin_d_sum_by_columns_mpi::SumColumnsMatrixMPI::PreProcessingImpl() {
  // init data
  if (world_.rank() == 0) {
    auto *ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    input_matrix_.assign(ptr, ptr + (rows_ * cols_));
    output_.resize(cols_, 0.0);
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
  int rank = world_.rank();
  int size = world_.size();
  boost::mpi::broadcast(world_, rows_, 0);
  boost::mpi::broadcast(world_, cols_, 0);
  size_t proc_count = static_cast<size_t>(size);
  size_t local_rows = rows_ / proc_count + (static_cast<size_t>(rank) < (rows_ % proc_count) ? 1 : 0);
  std::vector<int> local_matrix(local_rows * cols_);

  std::vector<int> send_counts;
  std::vector<int> displs;

  if (rank == 0) {
    send_counts.resize(size);
    displs.resize(size);
    size_t offset = 0;
    for (int i = 0; i < size; ++i) {
      size_t rows_for_proc = rows_ / proc_count + (static_cast<size_t>(i) < (rows_ % proc_count) ? 1 : 0);
      send_counts[i] = static_cast<int>(rows_for_proc * cols_);
      displs[i] = static_cast<int>(offset);
      offset += rows_for_proc * cols_;
    }
  }
  if (rank == 0) {
    boost::mpi::scatterv(world_, input_matrix_.data(), send_counts, displs, local_matrix.data(), 0);
  } else {
    boost::mpi::scatterv(world_, local_matrix.data(), 0);
  }

  std::vector<int> local_sum(cols_, 0);
  for (size_t row = 0; row < local_rows; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      local_sum[col] += local_matrix[row * cols_ + col];
    }
  }
  std::vector<int> gathered_sums;
  if (rank == 0) {
    gathered_sums.resize(size * cols_);
  }
  boost::mpi::gather(world_, local_sum.data(), static_cast<int>(cols_), gathered_sums.data(), 0);
  if (rank == 0) {
    output_.assign(cols_, 0);
    for (int proc = 0; proc < size; ++proc) {
      for (size_t col = 0; col < cols_; ++col) {
        output_[col] += gathered_sums[proc * cols_ + col];
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