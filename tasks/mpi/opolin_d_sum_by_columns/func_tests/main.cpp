// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

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

TEST(opolin_d_sum_by_columns_mpi, test_3x3_matrix) {
  boost::mpi::communicator world;
  size_t rows = 3;
  size_t cols = 3;

  std::vector<int> matrix;
  std::vector<int> expected;

  std::vector<int> out(cols, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    expected = {12, 15, 18};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_mpi->inputs_count.emplace_back(rows);
    task_data_mpi->inputs_count.emplace_back(cols);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }
  opolin_d_sum_by_columns_mpi::SumColumnsMatrixMPI test_task_parallel(task_data_mpi);

  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(opolin_d_sum_by_columns_mpi, test_5x1_matrix) {
  boost::mpi::communicator world;
  size_t rows = 5;
  size_t cols = 1;

  std::vector<int> matrix;
  std::vector<int> expected;

  std::vector<int> out(cols, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = {1, 2, 3, 4, 5};
    expected = {15};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_mpi->inputs_count.emplace_back(rows);
    task_data_mpi->inputs_count.emplace_back(cols);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }
  opolin_d_sum_by_columns_mpi::SumColumnsMatrixMPI test_task_parallel(task_data_mpi);

  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(opolin_d_sum_by_columns_mpi, test_1x5_matrix) {
  boost::mpi::communicator world;
  size_t rows = 1;
  size_t cols = 5;

  std::vector<int> matrix;
  std::vector<int> expected;

  std::vector<int> out(cols, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = {1, 2, 3, 4, 5};
    expected = {1, 2, 3, 4, 5};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_mpi->inputs_count.emplace_back(rows);
    task_data_mpi->inputs_count.emplace_back(cols);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }
  opolin_d_sum_by_columns_mpi::SumColumnsMatrixMPI test_task_parallel(task_data_mpi);

  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(opolin_d_sum_by_columns_mpi, test_single_element_matrix) {
  boost::mpi::communicator world;
  size_t rows = 1;
  size_t cols = 1;

  std::vector<int> matrix;
  std::vector<int> expected;

  std::vector<int> out(cols, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = {7};
    expected = {7};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_mpi->inputs_count.emplace_back(rows);
    task_data_mpi->inputs_count.emplace_back(cols);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }
  opolin_d_sum_by_columns_mpi::SumColumnsMatrixMPI test_task_parallel(task_data_mpi);

  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(opolin_d_sum_by_columns_mpi, test_negative_values) {
  boost::mpi::communicator world;
  size_t rows = 3;
  size_t cols = 3;

  std::vector<int> matrix;
  std::vector<int> expected;

  std::vector<int> out(cols, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = {-2, -4, -12, -9, -6, -1, -23, -7, -8};
    expected = {-34, -17, -21};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_mpi->inputs_count.emplace_back(rows);
    task_data_mpi->inputs_count.emplace_back(cols);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }
  opolin_d_sum_by_columns_mpi::SumColumnsMatrixMPI test_task_parallel(task_data_mpi);

  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(opolin_d_sum_by_columns_mpi, test_wrong_size) {
  boost::mpi::communicator world;
  size_t rows = 3;
  size_t cols = 0;

  std::vector<int> matrix;

  std::vector<int> out(cols, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_mpi->inputs_count.emplace_back(rows);
    task_data_mpi->inputs_count.emplace_back(cols);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }
  opolin_d_sum_by_columns_mpi::SumColumnsMatrixMPI test_task_parallel(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_parallel.Validation(), false);
  }
}

TEST(opolin_d_sum_by_columns_mpi, test_100x100_matrix) {
  boost::mpi::communicator world;
  size_t rows = 100;
  size_t cols = 100;

  std::vector<int> matrix;
  std::vector<int> expected;

  std::vector<int> out(cols, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    opolin_d_sum_by_columns_mpi::GenerateTestData(rows, cols, matrix, expected);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_mpi->inputs_count.emplace_back(rows);
    task_data_mpi->inputs_count.emplace_back(cols);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }
  opolin_d_sum_by_columns_mpi::SumColumnsMatrixMPI test_task_parallel(task_data_mpi);

  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(opolin_d_sum_by_columns_mpi, test_5x2_matrix) {
  boost::mpi::communicator world;
  size_t rows = 5;
  size_t cols = 2;

  std::vector<int> matrix;
  std::vector<int> expected;

  std::vector<int> out(cols, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = {2, 12, 5, -7, 1, 8, 21, 9, -15, 6};
    expected = {14, 28};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_mpi->inputs_count.emplace_back(rows);
    task_data_mpi->inputs_count.emplace_back(cols);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }
  opolin_d_sum_by_columns_mpi::SumColumnsMatrixMPI test_task_parallel(task_data_mpi);

  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(opolin_d_sum_by_columns_mpi, test_2x5_matrix) {
  boost::mpi::communicator world;
  size_t rows = 2;
  size_t cols = 5;

  std::vector<int> matrix;
  std::vector<int> expected;

  std::vector<int> out(cols, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = {2, 12, 5, -7, 1, 8, 21, 9, -15, 6};
    expected = {10, 33, 14, -22, 7};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_mpi->inputs_count.emplace_back(rows);
    task_data_mpi->inputs_count.emplace_back(cols);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }
  opolin_d_sum_by_columns_mpi::SumColumnsMatrixMPI test_task_parallel(task_data_mpi);

  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(opolin_d_sum_by_columns_mpi, test_simple_matrix) {
  boost::mpi::communicator world;
  size_t rows = 3;
  size_t cols = 3;

  std::vector<int> matrix;
  std::vector<int> expected;

  std::vector<int> out(cols, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    expected = {1, 1, 1};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_mpi->inputs_count.emplace_back(rows);
    task_data_mpi->inputs_count.emplace_back(cols);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }
  opolin_d_sum_by_columns_mpi::SumColumnsMatrixMPI test_task_parallel(task_data_mpi);

  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}