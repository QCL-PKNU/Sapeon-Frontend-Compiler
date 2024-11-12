#include "calibration/histogram.hpp"

#include <cstdlib>
#include <numeric>
#include <vector>

#include "calibration/max.hpp"
#include "glog/logging.h"

namespace spgraph_simulator {
namespace calibration {
Histogram::Histogram(const size_t num_bins) : num_bins_(num_bins) {
  histogram_edge_ = std::vector<float>(num_bins + 1);
  histogram_ = std::vector<size_t>(num_bins);
}

void Histogram::SetRange(const Tensor &tensor) {
  auto max = Max{};

  // TODO: Check Error
  max.Set(tensor);

  auto max_val = max.value();
  SetRange(max_val);
}

void Histogram::SetRange(const float range) {
  const int num_histogram_edge = histogram_edge_.size();
  const int num_histogram_bins = num_histogram_edge - 1;
  const float step_val = range * 2 / static_cast<float>(num_histogram_bins);

#pragma omp parallel for simd schedule(static) default(shared)
  for (int i = 0; i < num_histogram_edge; ++i) {
    histogram_edge_[i] = -range + step_val * static_cast<float>(i);
  }
}

void Histogram::Collect(const Tensor &tensor) {
  // If not initialized, initialize histogram_edge_ using range from first image
  if (!IsHistogramInitialized()) {
    SetRange(tensor);
  }

  const float *datas = tensor.data<float>();
  const auto &dimension = tensor.dimension();
  const int num_histogram_bins = histogram_.size();

  const int num_n = dimension.n();
  const int num_c = dimension.c();
  const int num_h = dimension.h();
  const int num_w = dimension.w();
  const size_t offset_w = 1;
  const size_t offset_h = num_w * offset_w;
  const size_t offset_c = num_h * offset_h;
  const size_t offset_n = num_c * offset_c;
  const size_t total_count = offset_n * num_n;

  std::unique_ptr<size_t[]> histogram_data(new size_t[total_count]);

#pragma omp parallel for schedule(static) default(shared) collapse(4)
  for (int n = 0; n < num_n; ++n) {
    for (int c = 0; c < num_c; ++c) {
      for (int h = 0; h < num_h; ++h) {
        for (int w = 0; w < num_w; ++w) {
          const float value =
              datas[n * offset_n + c * offset_c + h * offset_h + w];
          histogram_data[n * offset_n + c * offset_c + h * offset_h + w] =
              Stack(value);
        }
      }
    }
  }

  for (int n = 0; n < total_count; ++n) {
    histogram_.at(histogram_data[n])++;
  }
}

size_t Histogram::Stack(float value) noexcept {
  auto num_histogram_bins = histogram_.size();

  int start = 0;
  int end = num_histogram_bins - 1;
  while (start < end) {
    int middle = (start + end) / 2;
    if (value < histogram_edge_.at(middle)) {
      end = middle;
    } else if (middle + 1 < num_histogram_bins &&
               value >= histogram_edge_.at(middle + 1)) {
      start = middle + 1;
    } else {
      return middle;
      break;
    }

    if (start == end) {
      return middle + 1;
    }
  }

  return 0;
}

size_t Histogram::TotalCount() const noexcept {
  const size_t total_count = std::accumulate(
      histogram_.begin(), histogram_.end(), static_cast<size_t>(0));
  return total_count;
}

bool Histogram::IsHistogramInitialized() noexcept {
  if (histogram_edge_.size() > 0 && histogram_edge_.at(0) != 0 &&
      histogram_edge_.at(histogram_edge_.size() - 1) != 0) {
    return true;
  }
  if (TotalCount() > 0) {
    return true;
  }
  return false;
}
}  // namespace calibration
}  // namespace spgraph_simulator
