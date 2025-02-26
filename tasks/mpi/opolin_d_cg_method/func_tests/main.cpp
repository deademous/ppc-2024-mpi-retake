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
#include "mpi/opolin_d_cg_method/include/ops_mpi.hpp"

namespace opolin_d_cg_method_mpi {
namespace {
void GenDataCgMethod(size_t size, std::vector<double> &a, std::vector<double> &b, std::vector<double> &expected) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dist(-5.0, 5.0);
  std::vector<double> m(size * size);
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      m[(i * size) + j] = dist(gen);
    }
  }
  a.assign(size * size, 0.0);
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      for (size_t k = 0; k < size; k++) {
        a[(i * size) + j] += m[(k * size) + i] * m[(k * size) + j];
      }
    }
  }
  for (size_t i = 0; i < size; i++) {
    a[(i * size) + i] += static_cast<double>(size);
  }
  expected.resize(size);
  for (size_t i = 0; i < size; i++) {
    expected[i] = dist(gen);
  }
  b.assign(size, 0.0);
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      b[i] += a[(i * size) + j] * expected[j];
    }
  }
}
}  // namespace
}  // namespace opolin_d_cg_method_mpi

TEST(opolin_d_cg_method_mpi, test_small_system) {
  boost::mpi::communicator world;
  int size = 5;
  double epsilon = 1e-8;

  std::vector<double> x_ref;
  std::vector<double> a;
  std::vector<double> b;

  std::vector<double> x_out(size, 0.0);
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    opolin_d_cg_method_mpi::GenDataCgMethod(size, a, b, x_ref);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  opolin_d_cg_method_mpi::CGMethodkMPI test_task_parallel(task_data_mpi);

  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }
}

TEST(opolin_d_cg_method_mpi, test_big_system) {
  boost::mpi::communicator world;
  int size = 5;
  double epsilon = 1e-8;

  std::vector<double> x_ref;
  std::vector<double> a;
  std::vector<double> b;

  std::vector<double> x_out(size, 0.0);
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    opolin_d_cg_method_mpi::GenDataCgMethod(size, a, b, x_ref);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  opolin_d_cg_method_mpi::CGMethodkMPI test_task_parallel(task_data_mpi);

  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }
}

TEST(opolin_d_cg_method_mpi, test_correct_input) {
  boost::mpi::communicator world;
  int size = 3;
  double epsilon = 1e-8;

  std::vector<double> x_ref;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> x_out(size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    a = {4.0, 1.0, 2.0, 1.0, 5.0, 1.0, 2.0, 1.0, 5.0};
    b = {7.0, 7.0, 8.0};
    x_ref = {1.0, 1.0, 1.0};

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  opolin_d_cg_method_mpi::CGMethodkMPI test_task_parallel(task_data_mpi);

  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }
}

TEST(opolin_d_cg_method_mpi, test_no_simetric_matrix) {
  boost::mpi::communicator world;
  int size = 3;
  double epsilon = 1e-8;
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<double> a = {29.0, 0.0, 39.0, 29.0, 53.0, 17.0, 39.0, 1.0, 90.0};
    std::vector<double> b = {0.0, 0.0, 0.0};

    std::vector<double> x_out(size, 0.0);

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());

    opolin_d_cg_method_mpi::CGMethodkMPI test_task_parallel(task_data_mpi);
    ASSERT_EQ(test_task_parallel.Validation(), false);
  }
}

TEST(opolin_d_cg_method_mpi, test_negative_values) {
  boost::mpi::communicator world;
  int size = 3;
  double epsilon = 1e-8;

  std::vector<double> x_ref;
  std::vector<double> a;
  std::vector<double> b;

  std::vector<double> x_out(size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    a = {244.913, -64.084, 59.893, -64.084, 84.215, -23.392, 59.893, -23.392, 31.227};
    b = {47.955, -146.484, 35.406};
    x_ref = {-0.437926, -1.924931, 0.531806};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  opolin_d_cg_method_mpi::CGMethodkMPI test_task_parallel(task_data_mpi);

  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }
}

TEST(opolin_d_cg_method_mpi, test_no_positive_define_matrix) {
  boost::mpi::communicator world;
  int size = 3;
  double epsilon = 1e-8;

  std::vector<double> a;
  std::vector<double> b;

  std::vector<double> x_out(size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    a = {0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0};
    b = {0.0, 0.0, 0.0};

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());

    opolin_d_cg_method_mpi::CGMethodkMPI test_task_parallel(task_data_mpi);
    ASSERT_EQ(test_task_parallel.Validation(), false);
  }
}

TEST(opolin_d_cg_method_mpi, test_simple_matrix) {
  boost::mpi::communicator world;
  int size = 3;
  double epsilon = 1e-8;

  std::vector<double> x_ref;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> x_out(size, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    a = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    b = {1.0, 1.0, 1.0};
    x_ref = {1.0, 1.0, 1.0};

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  opolin_d_cg_method_mpi::CGMethodkMPI test_task_parallel(task_data_mpi);

  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }
}

TEST(opolin_d_cg_method_mpi, test_single_element) {
  boost::mpi::communicator world;
  int size = 1;
  double epsilon = 1e-5;

  std::vector<double> x_ref;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> x_out(size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    a = {1.0};
    b = {10.0};
    x_ref = {10.0};

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  opolin_d_cg_method_mpi::CGMethodkMPI test_task_parallel(task_data_mpi);

  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }
}