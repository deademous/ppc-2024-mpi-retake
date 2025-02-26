// Copyright 2023 Nesterov Alexander
#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace opolin_d_cg_method_seq {

bool IsPositiveDefinite(const std::vector<double>& mat, size_t size);
bool IsSimmetric(const std::vector<double>& mat, size_t size);
double ScalarProduct(const std::vector<double>& a, const std::vector<double>& b);
std::vector<double> MultiplyVecMat(const std::vector<double>& vec, const std::vector<double>& mat);

class CGMethodSequential : public ppc::core::Task {
 public:
  explicit CGMethodSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> A_;
  std::vector<double> b_;
  std::vector<double> x_;
  size_t n_;
  double epsilon_;
};

}  // namespace opolin_d_cg_method_seq