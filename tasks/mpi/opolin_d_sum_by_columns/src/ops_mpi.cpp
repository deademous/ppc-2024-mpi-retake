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
    auto *ptr = reinterpret_cast<double *>(task_data->inputs[0]);
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
  int local_rows = rows_ / size + (rank < (rows_ % size) ? 1 : 0);
  std::vector<double> local_matrix(local_rows * cols_);
  std::vector<int> send_counts;
  std::vector<int> displs;
  if (rank == 0) {
    send_counts.resize(size);
    displs.resize(size);
    int offset = 0;
    for (int i = 0; i < size; ++i) {
      int rows_for_proc = rows_ / size + (i < (rows_ % size) ? 1 : 0);
      send_counts[i] = rows_for_proc * cols_;
      displs[i] = offset;
      offset += rows_for_proc * cols_;
    }
  }
  boost::mpi::scatterv(world_, input_matrix_, send_counts, displs, local_matrix, 0);
  std::vector<double> local_sum(cols_, 0.0);
  for (int row = 0; row < local_rows; ++row) {
    for (int col = 0; col < cols_; ++col) {
      local_sum[col] += local_matrix[row * cols_ + col];
    }
  }
  std::vector<double> gathered_sums;
  boost::mpi::gather(world_, local_sum, gathered_sums, 0);
  if (rank == 0) {
    output_.assign(cols_, 0.0);
    for (int proc = 0; proc < size; ++proc) {
      for (int col = 0; col < cols_; ++col) {
        output_[col] += gathered_sums[proc * cols_ + col];
      }
    }
  }
  return true;
}

bool opolin_d_sum_by_columns_mpi::SumColumnsMatrixMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto *out = reinterpret_cast<double *>(task_data->outputs[0]);
    std::ranges::copy(output_, out);
  }
  return true;
}