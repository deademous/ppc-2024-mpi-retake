// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/opolin_d_sum_by_columns/include/ops_seq.hpp"

namespace opolin_d_sum_by_columns_seq {
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
}  // namespace opolin_d_sum_by_columns_seq

TEST(opolin_d_sum_by_columns_seq, test_3x3_matrix) {
  size_t rows = 3;
  size_t cols = 3;
  std::vector<int> matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> expected = {12, 15, 18};

  std::vector<int> out(cols, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  opolin_d_sum_by_columns_seq::SumColumnsMatrixSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(opolin_d_sum_by_columns_seq, test_5x1_matrix) {
  size_t rows = 5;
  size_t cols = 1;
  std::vector<int> matrix = {1, 2, 3, 4, 5};
  std::vector<int> expected = {15};

  std::vector<int> out(cols, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  opolin_d_sum_by_columns_seq::SumColumnsMatrixSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(opolin_d_sum_by_columns_seq, test_1x5_matrix) {
  size_t rows = 1;
  size_t cols = 5;
  std::vector<int> matrix = {1, 2, 3, 4, 5};
  std::vector<int> expected = {1, 2, 3, 4, 5};

  std::vector<int> out(cols, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  opolin_d_sum_by_columns_seq::SumColumnsMatrixSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(opolin_d_sum_by_columns_seq, test_single_element_matrix) {
  size_t rows = 1;
  size_t cols = 1;
  std::vector<int> matrix = {7};
  std::vector<int> expected = {7};

  std::vector<int> out(cols, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  opolin_d_sum_by_columns_seq::SumColumnsMatrixSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(opolin_d_sum_by_columns_seq, test_negative_values) {
  size_t rows = 3;
  size_t cols = 3;
  std::vector<int> matrix = {-2, -4, -12, -9, -6, -1, -23, -7, -8};
  std::vector<int> expected = {-34, -17, -21};

  std::vector<int> out(cols, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  opolin_d_sum_by_columns_seq::SumColumnsMatrixSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(opolin_d_sum_by_columns_seq, test_wrong_size) {
  size_t rows = 0;
  size_t cols = 3;
  std::vector<int> matrix;

  std::vector<int> out(cols, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  opolin_d_sum_by_columns_seq::SumColumnsMatrixSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(opolin_d_sum_by_columns_seq, test_100x100_matrix) {
  size_t rows = 100;
  size_t cols = 100;
  std::vector<int> matrix;
  std::vector<int> expected;

  std::vector<int> out(cols, 0);
  opolin_d_sum_by_columns_seq::GenerateTestData(rows, cols, matrix, expected);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  opolin_d_sum_by_columns_seq::SumColumnsMatrixSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(opolin_d_sum_by_columns_seq, test_5x2_matrix) {
  size_t rows = 5;
  size_t cols = 2;
  std::vector<int> matrix = {2, 12, 5, -7, 1, 8, 21, 9, -15, 6};
  std::vector<int> expected = {14, 28};

  std::vector<int> out(cols, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  opolin_d_sum_by_columns_seq::SumColumnsMatrixSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(opolin_d_sum_by_columns_seq, test_2x5_matrix) {
  size_t rows = 2;
  size_t cols = 5;
  std::vector<int> matrix = {2, 12, 5, -7, 1, 8, 21, 9, -15, 6};
  std::vector<int> expected = {10, 33, 14, -22, 7};

  std::vector<int> out(cols, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  opolin_d_sum_by_columns_seq::SumColumnsMatrixSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(opolin_d_sum_by_columns_seq, test_simple_matrix) {
  size_t rows = 3;
  size_t cols = 3;
  std::vector<int> matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<int> expected = {1, 1, 1};

  std::vector<int> out(cols, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  opolin_d_sum_by_columns_seq::SumColumnsMatrixSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  ASSERT_EQ(expected, out);
}