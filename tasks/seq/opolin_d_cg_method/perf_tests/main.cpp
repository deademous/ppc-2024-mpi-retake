// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/opolin_d_cg_method/include/ops_seq.hpp"

namespace opolin_d_cg_method_seq {
namespace {
void genDataCGMethod(size_t size, std::vector<double> &A, std::vector<double> &b, std::vector<double> &expectedX) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dist(-5.0, 5.0);
  std::vector<double> M(size * size);
  for (size_t i = 0; i < size; i++)
    for (size_t j = 0; j < size; j++) M[i * size + j] = dist(gen);

  A.assign(size * size, 0.0);
  for (size_t i = 0; i < size; i++)
    for (size_t j = 0; j < size; j++)
      for (size_t k = 0; k < size; k++) A[i * size + j] += M[k * size + i] * M[k * size + j];

  for (size_t i = 0; i < size; i++) A[i * size + i] += size;

  expectedX.resize(size);
  for (size_t i = 0; i < size; i++) expectedX[i] = dist(gen);

  b.assign(size, 0.0);
  for (size_t i = 0; i < size; i++)
    for (size_t j = 0; j < size; j++) b[i] += A[i * size + j] * expectedX[j];
}
}  // namespace
}  // namespace opolin_d_cg_method_seq

TEST(opolin_d_cg_method_seq, test_pipeline_run) {
  int size = 1500;
  std::vector<double> A;
  std::vector<double> b;
  std::vector<double> X;
  opolin_d_cg_method_seq::genDataCGMethod(size, A, b, X);
  std::vector<double> out(size, 0);
  double epsilon = 1e-7;
  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  task_data_seq->inputs_count.emplace_back(out.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential = std::make_shared<opolin_d_cg_method_seq::CGMethodSequential>(task_data_seq);
  ASSERT_EQ(test_task_sequential->Validation(), true);
  test_task_sequential->PreProcessing();
  test_task_sequential->Run();
  test_task_sequential->PostProcessing();
  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(opolin_d_cg_method_seq, test_task_run) {
  int size = 1500;
  std::vector<double> A;
  std::vector<double> b;
  std::vector<double> X;
  opolin_d_cg_method_seq::genDataCGMethod(size, A, b, X);
  std::vector<double> out(size, 0);
  double epsilon = 1e-7;
  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  task_data_seq->inputs_count.emplace_back(out.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential = std::make_shared<opolin_d_simple_iteration_method_seq::CGMethodSequential>(task_data_seq);
  ASSERT_EQ(test_task_sequential->Validation(), true);
  test_task_sequential->PreProcessing();
  test_task_sequential->Run();
  test_task_sequential->PostProcessing();
  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}