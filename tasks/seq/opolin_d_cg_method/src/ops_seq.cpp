// Copyright 2024 Nesterov Alexander
#include "seq/opolin_d_cg_method/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

using namespace std::chrono_literals;

bool opolin_d_cg_method_seq::CGMethodSequential::PreProcessingImpl() {
  auto* ptr = reinterpret_cast<double*>(task_data->inputs[1]);
  b_.assign(ptr, ptr + n_);

  epsilon_ = *reinterpret_cast<double*>(task_data->inputs[2]);
  return true;
}

bool opolin_d_cg_method_seq::CGMethodSequential::ValidationImpl() {
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

  if (!opolin_d_cg_method_seq::IsSimmetric(A_, n_)) {
    return false;
  }
  if (!opolin_d_cg_method_seq::IsPositiveDefinite(A_, n_)) {
    return false;
  }
  return true;
}

bool opolin_d_cg_method_seq::CGMethodSequential::RunImpl() {
  x_.resize(n_);
  std::vector<double> r_k = b_;
  std::vector<double> p_k = r_k;
  while (true) {
    double rsquare_prev = opolin_d_cg_method_seq::ScalarProduct(r_k, r_k);
    std::vector<double> ap = opolin_d_cg_method_seq::MultiplyVecMat(p_k, A_);
    double alpha_k = rsquare_prev / opolin_d_cg_method_seq::ScalarProduct(p_k, ap);

    // x_k+1
    for (int i = 0; i < static_cast<int>(n_); i++) {
      x_[i] += alpha_k * p_k[i];
    }

    // r_k+1
    for (int i = 0; i < static_cast<int>(n_); i++) {
      r_k[i] -= alpha_k * ap[i];
    }

    double rsquare_k = opolin_d_cg_method_seq::ScalarProduct(r_k, r_k);
    // right accuracy is achieved
    if (sqrt(rsquare_k) < epsilon_) {
      break;
    }

    double beta_k = rsquare_k / rsquare_prev;
    // p_k+1
    for (int i = 0; i < static_cast<int>(n_); i++) {
      p_k[i] = r_k[i] + beta_k * p_k[i];
    }
  }
  return true;
}

bool opolin_d_cg_method_seq::CGMethodSequential::PostProcessingImpl() {
  auto* out = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(x_, out);
  return true;
}

bool opolin_d_cg_method_seq::IsPositiveDefinite(const std::vector<double>& mat, size_t size) {
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

bool opolin_d_cg_method_seq::IsSimmetric(const std::vector<double>& mat, size_t size) {
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

double opolin_d_cg_method_seq::ScalarProduct(const std::vector<double>& a, const std::vector<double>& b) {
  size_t size = a.size();
  double result = 0.0;
  for (size_t i = 0; i < size; i++) {
    result += a[i] * b[i];
  }
  return result;
}

std::vector<double> opolin_d_cg_method_seq::MultiplyVecMat(const std::vector<double>& vec,
                                                           const std::vector<double>& mat) {
  size_t size = vec.size();
  std::vector<double> result(size, 0.0);
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      result[i] += mat[(i * size) + j] * vec[j];
    }
  }
  return result;
}