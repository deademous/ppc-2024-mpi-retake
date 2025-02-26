// Copyright 2024 Nesterov Alexander
#include "mpi/opolin_d_cg_method/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(misc-include-cleaner)
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

bool opolin_d_cg_method_mpi::CGMethodkMPI::PreProcessingImpl() {
  // init data
  if (world_.rank() == 0) {
    auto* ptr = reinterpret_cast<double*>(task_data->inputs[1]);
    b_.assign(ptr, ptr + n_);

    epsilon_ = *reinterpret_cast<double*>(task_data->inputs[2]);
  }
  return true;
}

bool opolin_d_cg_method_mpi::CGMethodkMPI::ValidationImpl() {
  // check input and output
  if (world_.rank() == 0) {
    if (task_data->inputs_count.empty() || task_data->inputs.size() != 3) {
      return false;
    }

    if (task_data->outputs_count.empty() || task_data->inputs_count[0] != task_data->outputs_count[0] ||
        task_data->outputs.empty()) {
      return false;
    }

    n_ = task_data->inputs_count[0];
    if (n_ <= 0) {
      return false;
    }
    auto* ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    A_.assign(ptr, ptr + (n_ * n_));

    if (!IsSimmetric(A_, n_)) {
      return false;
    }
    if (!IsPositiveDefinite(A_, n_)) {
      return false;
    }
  }
  return true;
}

bool opolin_d_cg_method_mpi::CGMethodkMPI::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  boost::mpi::broadcast(world_, n_, 0);
  boost::mpi::broadcast(world_, epsilon_, 0);
  size_t local_n = n_ / size;
  if (rank < static_cast<int>(n_ % size)) {
    ++local_n;
  }
  std::vector<int> send_counts;
  std::vector<int> displs;
  std::vector<int> send_counts_a;
  std::vector<int> displs_a;
  if (rank == 0) {
    send_counts.resize(size);
    displs.resize(size);
    send_counts_a.resize(size);
    displs_a.resize(size);

    size_t offset = 0;
    size_t offset_a = 0;
    for (int i = 0; i < size; ++i) {
      size_t rows = (n_ / size) + static_cast<size_t>(i < static_cast<int>(n_ % size));
      send_counts[i] = static_cast<int>(rows);
      displs[i] = static_cast<int>(offset);
      offset += rows;

      send_counts_a[i] = static_cast<int>(rows * n_);
      displs_a[i] = static_cast<int>(offset_a);
      offset_a += rows * n_;
    }
  }
  std::vector<double> local_a(local_n * n_);
  if (rank == 0) {
    boost::mpi::scatterv(world_, A_.data(), send_counts_a, displs_a, local_a.data(), static_cast<int>(local_n * n_), 0);
  } else {
    boost::mpi::scatterv(world_, local_a.data(), static_cast<int>(local_n * n_), 0);
  }
  std::vector<double> local_b(local_n);
  if (rank == 0) {
    boost::mpi::scatterv(world_, b_.data(), send_counts, displs, local_b.data(), static_cast<int>(local_n), 0);
  } else {
    boost::mpi::scatterv(world_, local_b.data(), static_cast<int>(local_n), 0);
  }
  std::vector<double> local_x(local_n, 0.0);
  std::vector<double> local_r = local_b;
  std::vector<double> local_p = local_r;
  std::vector<double> local_ap(local_n);
  std::vector<double> full_p(n_);

  double rsquare_prev = 0.0;
  while (true) {
    double local_rsquare = opolin_d_cg_method_mpi::ScalarProduct(local_r, local_r);
    double rsquare_k = 0.0;
    boost::mpi::reduce(world_, local_rsquare, rsquare_k, std::plus<>(), 0);
    boost::mpi::broadcast(world_, rsquare_k, 0);

    rsquare_prev = rsquare_k;
    if (rank == 0) {
      boost::mpi::gatherv(world_, local_p.data(), static_cast<int>(local_n), full_p.data(), send_counts, displs, 0);
    } else {
      boost::mpi::gatherv(world_, local_p.data(), static_cast<int>(local_n), 0);
    }
    boost::mpi::broadcast(world_, full_p, 0);

    for (size_t i = 0; i < local_n; ++i) {
      local_ap[i] = 0.0;
      for (size_t j = 0; j < n_; ++j) {
        local_ap[i] += local_a[(i * n_) + j] * full_p[j];
      }
    }

    // p^T * A * p
    double local_p_ap = opolin_d_cg_method_mpi::ScalarProduct(local_p, local_ap);
    double p_ap = 0.0;
    boost::mpi::reduce(world_, local_p_ap, p_ap, std::plus<>(), 0);
    boost::mpi::broadcast(world_, p_ap, 0);

    // alpha_k
    double alpha_k = rsquare_prev / p_ap;

    for (size_t i = 0; i < local_n; ++i) {
      // x_k+1
      local_x[i] += alpha_k * local_p[i];
      // r_k+1
      local_r[i] -= alpha_k * local_ap[i];
    }

    local_rsquare = opolin_d_cg_method_mpi::ScalarProduct(local_r, local_r);
    rsquare_k = 0.0;
    boost::mpi::reduce(world_, local_rsquare, rsquare_k, std::plus<>(), 0);
    boost::mpi::broadcast(world_, rsquare_k, 0);

    if (sqrt(rsquare_k) < epsilon_) {
      break;
    }
    double beta_k = rsquare_k / rsquare_prev;
    for (size_t i = 0; i < local_n; ++i) {
      local_p[i] = local_r[i] + beta_k * local_p[i];
    }
  }

  x_.resize(n_);
  boost::mpi::gatherv(world_, local_x.data(), static_cast<int>(local_n), x_.data(), send_counts, displs, 0);

  return true;
}

bool opolin_d_cg_method_mpi::CGMethodkMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto* out = reinterpret_cast<double*>(task_data->outputs[0]);
    std::ranges::copy(x_, out);
  }
  return true;
}

bool opolin_d_cg_method_mpi::IsPositiveDefinite(const std::vector<double>& mat, size_t size) {
  std::vector<double> l(size * size, 0);

  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j <= i; j++) {
      double sum = 0;
      if (j == i) {
        for (size_t k = 0; k < j; k++) {
          sum += l[(j * size) + k] * l[(j * size) + k];
        }
        double val = mat[(j * size) + j] - sum;
        if (val <= 0) {
          return false;
        }
        l[(j * size) + j] = std::sqrt(val);
      } else {
        for (size_t k = 0; k < j; k++) {
          sum += l[(i * size) + k] * l[(j * size) + k];
        }
        l[(i * size) + j] = (mat[(i * size) + j] - sum) / l[(j * size) + j];
      }
    }
  }
  return true;
}

bool opolin_d_cg_method_mpi::IsSimmetric(const std::vector<double>& mat, size_t size) {
  bool simetric = true;
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      if (j != i) {
        if (mat[(i * size) + j] != mat[(j * size) + i]) {
          simetric = false;
        }
      }
    }
  }
  return simetric;
}

double opolin_d_cg_method_mpi::ScalarProduct(const std::vector<double>& a, const std::vector<double>& b) {
  size_t size = a.size();
  double result = 0.0;
  for (size_t i = 0; i < size; i++) {
    result += a[i] * b[i];
  }
  return result;
}