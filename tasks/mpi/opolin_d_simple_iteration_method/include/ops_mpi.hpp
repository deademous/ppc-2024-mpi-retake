// Copyright 2024 Nesterov Alexander
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace opolin_d_simple_iteration_method_mpi {

size_t rank(std::vector<double> matrix, size_t n);
bool isDiagonalDominance(std::vector<double> mat, size_t dim);

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> A_;
  std::vector<double> C_;
  std::vector<double> b_;
  std::vector<double> d_;
  std::vector<double> Xold_;
  std::vector<double> Xnew_;
  uint32_t n_;
  double epsilon_;
  int max_iters_;
  boost::mpi::communicator world_;
};

}  // namespace opolin_d_simple_iteration_method_mpi