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
void genDataCGMethod(size_t size, std::vector<double>& A, std::vector<double>& b, std::vector<double>& expectedX) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dist(-5.0 , 5.0);

  std::vector<double> M(size * size);
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      M[i * size + j] = dist(gen);

  A.assign(size * size, 0.0);
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      for (int k = 0; k < size; k++)
        A[i * size + j] += M[k * size + i] * M[k * size + j];
    
  for (int i = 0; i < size; i++)
    A[i * size + i] += size;

  expectedX.resize(size);
  for (int i = 0; i < size; i++)
    expectedX[i] = dist(gen);

  b.assign(size, 0.0);
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      b[i] += A[i * size + j] * expectedX[j];
}
}  // namespace
}  // namespace opolin_d_cg_method_seq

TEST(opolin_d_cg_method_seq, test_small_system) {
  int size = 3;
  double epsilon = 1e-9;
  std::vector<double> expectedX, A, b;
  opolin_d_cg_method_seq::genDataCGMethod(size, A, b, expectedX);

  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  task_data_seq->inputs_count.emplace_back(out.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<opolin_d_cg_method_seq::CGMethodSequential>(task_data_seq);
  ASSERT_EQ(test_task_sequential->Validation(), true);
  test_task_sequential->PreProcessing();
  test_task_sequential->Run();
  test_task_sequential->PostProcessing();
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(expectedX[i], out[i], 1e-3);
  }
}

TEST(opolin_d_cg_method_seq, test_big_system) {
  int size = 100;
  double epsilon = 1e-9;
  std::vector<double> expectedX, A, b;
  opolin_d_cg_method_seq::genDataCGMethod(size, A, b, expectedX);

  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  task_data_seq->inputs_count.emplace_back(out.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<opolin_d_cg_method_seq::CGMethodSequential>(task_data_seq);
  ASSERT_EQ(test_task_sequential->Validation(), true);
  test_task_sequential->PreProcessing();
  test_task_sequential->Run();
  test_task_sequential->PostProcessing();
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(expectedX[i], out[i], 1e-3);
  }
}

TEST(opolin_d_cg_method_seq, test_correct_input) {
  int size = 3;
  double epsilon = 1e-9;
  std::vector<double> expectedX, A, b;
  A = {29.0, 29.0, 39.0, 29.0, 53.0, 17.0, 39.0, 17.0, 90.0};
  b = {204.0, 186.0, 343.0};
  expectedX = {1.0, 2.0, 3.0};
  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  task_data_seq->inputs_count.emplace_back(out.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<opolin_d_cg_method_seq::CGMethodSequential>(task_data_seq);
  ASSERT_EQ(test_task_sequential->Validation(), true);
  test_task_sequential->PreProcessing();
  test_task_sequential->Run();
  test_task_sequential->PostProcessing();
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(expectedX[i], out[i], 1e-3);
  }
}

TEST(opolin_d_cg_method_seq, test_no_simertric) {
  int size = 3;
  double epsilon = 1e-9;
  std::vector<double> A, b;
  A = {29.0, 0.0, 39.0, 29.0, 53.0, 17.0, 39.0, 1.0, 90.0};
  b = {0.0, 0.0, 0.0};
  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  task_data_seq->inputs_count.emplace_back(out.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<opolin_d_cg_method_seq::CGMethodSequential>(task_data_seq);
  ASSERT_EQ(test_task_sequential->Validation(), false);
}

TEST(opolin_d_cg_method_seq, test_no_positive_define) {
  int size = 3;
  double epsilon = 1e-9;
  std::vector<double> A, b;
  A = {0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0};
  b = {0.0, 0.0, 0.0};
  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  task_data_seq->inputs_count.emplace_back(out.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<opolin_d_cg_method_seq::CGMethodSequential>(task_data_seq);
  ASSERT_EQ(test_task_sequential->Validation(), false);
}

TEST(opolin_d_cg_method_seq, test_negative_values) {
  int size = 3;
  double epsilon = 1e-9;
  std::vector<double> expectedX, A, b;
  A = {244.913, -64.084, 59.893, -64.084, 84.215, -23.392, 59.893, -23.392, 31.227};
  b = {47.955, -146.484, 35.406};
  expectedX = {-0.437926, -1.924931, 0.531806};
  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  task_data_seq->inputs_count.emplace_back(out.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<opolin_d_cg_method_seq::CGMethodSequential>(task_data_seq);
  ASSERT_EQ(test_task_sequential->Validation(), true);
  test_task_sequential->PreProcessing();
  test_task_sequential->Run();
  test_task_sequential->PostProcessing();
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(expectedX[i], out[i], 1e-3);
  }
}

TEST(opolin_d_cg_method_seq, test_simple_element) {
  int size = 1;
  double epsilon = 1e-9;
  std::vector<double> expectedX, A, b;
  genDataCGMethod(size, A, b, expectedX);

  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  task_data_seq->inputs_count.emplace_back(out.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<opolin_d_cg_method_seq::CGMethodSequential>(task_data_seq);
  ASSERT_EQ(test_task_sequential->Validation(), true);
  test_task_sequential->PreProcessing();
  test_task_sequential->Run();
  test_task_sequential->PostProcessing();
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(expectedX[i], out[i], 1e-3);
  }
}

TEST(opolin_d_cg_method_seq, test_simple_matrix) {
  int size = 3;
  double epsilon = 1e-9;
  std::vector<double> expectedX, A, b;
  A = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  b = {1.0, 2.0, 3.0};
  expectedX = {1.0, 2.0, 3.0};
  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  task_data_seq->inputs_count.emplace_back(out.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<opolin_d_cg_method_seq::CGMethodSequential>(task_data_seq);
  ASSERT_EQ(test_task_sequential->Validation(), true);
  test_task_sequential->PreProcessing();
  test_task_sequential->Run();
  test_task_sequential->PostProcessing();
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(expectedX[i], out[i], 1e-3);
  }
}