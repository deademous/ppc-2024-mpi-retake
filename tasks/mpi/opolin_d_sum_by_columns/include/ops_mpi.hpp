// Copyright 2024 Nesterov Alexander
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace opolin_d_sum_by_columns_mpi {

class SumColumnsMatrixMPI : public ppc::core::Task {
 public:
  explicit SumColumnsMatrixMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_matrix_;
  std::vector<int> output_;
  size_t rows_;
  size_t cols_;
  boost::mpi::communicator world_;
};

}  // namespace opolin_d_sum_by_columns_mpi