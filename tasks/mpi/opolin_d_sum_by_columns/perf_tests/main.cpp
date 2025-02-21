// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/opolin_d_sum_by_columns/include/ops_mpi.hpp"

namespace opolin_d_sum_by_columns_mpi {
namespace {
void GenerateTestData(size_t rows, size_t cols, std::vector<int> &matrix, std::vector<int> &expected) {
  std::random_device dev;
  std::mt19937 gen(dev());
  expected.resize(cols, 0);
  matrix.resize(rows * cols);
  for (size_t i = 0; i < rows * cols; ++i) {
    matrix[i] = (static_cast<int>(gen()) % 200) - 100;
  }
  for (size_t col = 0; col < cols; ++col) {
    for (size_t row = 0; row < rows; ++row) {
      expected[col] += matrix[(row * cols) + col];
    }
  }
}
}  // namespace
}  // namespace opolin_d_sum_by_columns_mpi

TEST(opolin_d_sum_by_columns_mpi, test_pipeline_run) {
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  size_t rows = 4000;
  size_t cols = 4000;
  std::vector<int> matrix;
  std::vector<int> expected;

  std::vector<int> out(cols, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  // Create data
  if (world.rank() == 0) {
    opolin_d_sum_by_columns_mpi::GenerateTestData(rows, cols, matrix, expected);
    // Create TaskData
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_mpi->inputs_count.emplace_back(rows);
    task_data_mpi->inputs_count.emplace_back(cols);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  auto test_task_mpi = std::make_shared<opolin_d_sum_by_columns_mpi::SumColumnsMatrixMPI>(task_data_mpi);
  ASSERT_EQ(test_task_mpi->Validation(), true);
  test_task_mpi->PreProcessing();
  test_task_mpi->Run();
  test_task_mpi->PostProcessing();
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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  // Create Perf analyzer
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}

TEST(opolin_d_sum_by_columns_mpi, test_task_run) {
  boost::mpi::communicator world;
  size_t rows = 4000;
  size_t cols = 4000;
  std::vector<int> matrix;
  std::vector<int> expected;
  std::vector<int> out(cols, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    opolin_d_sum_by_columns_mpi::GenerateTestData(rows, cols, matrix, expected);
    // Create TaskData
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_mpi->inputs_count.emplace_back(rows);
    task_data_mpi->inputs_count.emplace_back(cols);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }
  auto test_task_mpi = std::make_shared<opolin_d_sum_by_columns_mpi::SumColumnsMatrixMPI>(task_data_mpi);
  ASSERT_EQ(test_task_mpi->Validation(), true);
  test_task_mpi->PreProcessing();
  test_task_mpi->Run();
  test_task_mpi->PostProcessing();
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  // Create Perf attributes
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  // Create Perf analyzer
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}