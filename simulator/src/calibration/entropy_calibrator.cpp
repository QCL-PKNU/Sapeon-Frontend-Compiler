#include "calibration/entropy_calibrator.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <optional>
#include <tuple>
#include <vector>

#include "calibration/histogram_calibrator.hpp"
#include "enums/error.hpp"
#include "glog/logging.h"
#include "network/tensor.hpp"
#include "tl/expected.hpp"

namespace spgraph_simulator {
namespace calibration {
std::unique_ptr<Calibrator> EntropyCalibrator::CreateInstance(
    Network& network, std::optional<float> percentile) {
  return std::make_unique<EntropyCalibrator>(network);
}

EntropyCalibrator::EntropyCalibrator(Network& network)
    : HistogramCalibrator(network) {
  const size_t num_layers = network_.num_layers();
  histograms_.reserve(num_layers);
  for (int i = 0; i < num_layers; i++) {
    // if (CheckSensitiveLayer(network_.layers(i))) {
    //   histograms_.push_back({80001});
    // } else {
    //   histograms_.push_back({8001});
    // }
    histograms_.push_back({8001});
  }
}

bool EntropyCalibrator::CheckSensitiveLayer(const Layer& layer) {
  const auto& operation_types = layer.operation_types();
  if (!layer.HasFilter()) {
    return false;
  }

  auto& dims = layer.filter()->dimension();
  if (dims.w() != 1 || dims.h() != 1) {
    return false;
  }

  for (const auto& type : operation_types) {
    if (type == "Convolution" || type == "GroupConvolution") {
      return true;
    }
  }

  return false;
}

size_t EntropyCalibrator::GetNumQuantizedHistogramBins(int num_histogram_bins) {
  const size_t kInt8QuantizedHistogramBins = 255;
  const size_t kInt16QuantizedHistogramBins = 65535;
  switch (num_histogram_bins) {
    case 8001:
      return kInt8QuantizedHistogramBins;
    case 80001:
      return kInt16QuantizedHistogramBins;
    default:
      LOG(WARNING) << "num_histogram_bins = " << num_histogram_bins;
      return kInt8QuantizedHistogramBins;
  }
}

tl::expected<void, SimulatorError> EntropyCalibrator::SmoothDistribution(
    std::vector<float>& vec, float epsilon) {
  int size_vec = vec.size();
  int num_zeros = 0;
  for (int i = 0; i < size_vec; ++i) {
    if (vec[i] == 0) {
      num_zeros++;
    }
  }
  int num_nonzeros = size_vec - num_zeros;
  if (num_nonzeros == 0) {
    LOG(ERROR) << "Vector's entries are all zero : malfunctioned vector";
    return tl::make_unexpected(SimulatorError::kCalibrationError);
  }

  const float normalized_epsilon = epsilon * static_cast<float>(num_zeros) /
                                   static_cast<float>(num_nonzeros);
  if (normalized_epsilon >= 1.0f) {
    LOG(ERROR) << "Normalized epsilon should be lower than 1.0f";
    return tl::make_unexpected(SimulatorError::kCalibrationError);
  }

  for (int i = 0; i < size_vec; ++i) {
    if (vec[i] == 0) {
      vec[i] = epsilon;
    } else {
      vec[i] = vec[i] - normalized_epsilon;
      if (vec[i] < 0) {
        LOG(ERROR) << "Epsilon is too big";
        return tl::make_unexpected(SimulatorError::kCalibrationError);
      }
    }
  }
  return {};
}

float EntropyCalibrator::CalculateEntropy(
    const std::vector<float>& original_vec,
    const std::vector<float>& quantized_vec) {
  float sum = 0.0f;
  int size = original_vec.size();
  for (int i = 0; i < size; ++i) {
    sum += original_vec[i] *
           std::log(static_cast<double>(original_vec[i] / quantized_vec[i]));
  }
  return sum;
}

std::unique_ptr<Calibrator::CalibrationResult>
EntropyCalibrator::ComputeRange() {
  const size_t num_layers = network_.num_layers();
  auto result = std::make_unique<Calibrator::CalibrationResult>();
  result->ranges.reserve(num_layers + 1);

#pragma omp parallel for collapse(1)
  for (int idx_layer = 0; idx_layer < num_layers; idx_layer++) {
    DLOG(INFO) << "Compute " << idx_layer << " layer";
    const auto& histogram = histograms_.at(idx_layer);
    const size_t num_total_data = histogram.TotalCount();

    DLOG_IF(FATAL, num_total_data == 0) << "histogram is empty";

    const size_t num_histogram_bins = histogram.num_bins();
    const size_t num_half_histogram_bins = num_histogram_bins / 2;

    const size_t num_quantized_histogram_bins =
        GetNumQuantizedHistogramBins(num_histogram_bins);
    const size_t num_half_quantized_histogram_bins =
        num_quantized_histogram_bins / 2;
    const size_t num_entropies =
        num_half_histogram_bins - num_half_quantized_histogram_bins + 1;

    std::vector<float> entropies(num_entropies);
    std::vector<size_t> quantized_histogram(num_quantized_histogram_bins);

    for (int i = num_half_quantized_histogram_bins;
         i < num_half_histogram_bins + 1; ++i) {
      const int idx_left = num_half_histogram_bins - i;
      const int idx_right = num_half_histogram_bins + i + 1;
      const int range = idx_right - idx_left;
      const auto num_data = std::accumulate(
          histogram.histogram().begin() + idx_left,
          histogram.histogram().begin() + idx_right, static_cast<size_t>(0));
      if (num_data == 0) {
        entropies[i - num_half_quantized_histogram_bins] =
            std::numeric_limits<float>::max();
        quantized_histogram.assign(quantized_histogram.size(), 0);
        continue;
      }
      auto partial_histogram =
          std::vector<float>(histogram.histogram().begin() + idx_left,
                             histogram.histogram().begin() + idx_right);
      auto clipped_partial_histogram = std::vector<float>(partial_histogram);
      size_t num_left_outlier = 0;

      for (int j = 0; j < idx_left; ++j) {
        num_left_outlier += histogram.histogram(j);
      }
      size_t num_right_outlier = 0;
      for (int j = idx_right; j < num_histogram_bins; ++j) {
        num_right_outlier += histogram.histogram(j);
      }
      clipped_partial_histogram.front() += num_left_outlier;
      clipped_partial_histogram.back() += num_right_outlier;
      auto is_nonzeros = std::vector<int>(range);
      for (int j = 0; j < range; ++j) {
        if (partial_histogram[j] > 0) {
          is_nonzeros[j] = 1;
        }
      }

      const int size_merged_range = range / num_quantized_histogram_bins;
      for (int j = 0; j < num_quantized_histogram_bins; ++j) {
        const int idx_start = j * size_merged_range;
        const int idx_end = idx_start + size_merged_range;
        size_t sum = 0;
        for (int k = idx_start; k < idx_end; ++k) {
          sum += partial_histogram[k];
        }
        quantized_histogram[j] = sum;
      }
      for (int j = num_quantized_histogram_bins * size_merged_range; j < range;
           ++j) {
        quantized_histogram.back() += partial_histogram[j];
      }

      auto quantized_partial_histogram = std::vector<float>(range);

      for (int j = 0; j < num_quantized_histogram_bins; ++j) {
        const int idx_start = j * size_merged_range;
        int idx_end = idx_start + size_merged_range;
        if (j == num_quantized_histogram_bins - 1) {
          idx_end = range;
        }
        int nonzero_count = 0;
        for (int k = idx_start; k < idx_end; ++k) {
          nonzero_count += is_nonzeros[k];
        }
        if (nonzero_count != 0) {
          for (int k = idx_start; k < idx_end; ++k) {
            quantized_partial_histogram[k] =
                static_cast<float>(quantized_histogram[j]) /
                static_cast<float>(nonzero_count);
            if (partial_histogram[k] == 0) {
              quantized_partial_histogram[k] = 0;
            }
          }
        }
      }
      {
        auto result = SmoothDistribution(quantized_partial_histogram, 0.0001);
        // if (!result) return make_unexpected(result.error());
      }
      {
        auto result = SmoothDistribution(clipped_partial_histogram, 0.0001);
        // if (!result) return make_unexpected(result.error());
      }
      entropies[i - num_half_quantized_histogram_bins] = CalculateEntropy(
          clipped_partial_histogram, quantized_partial_histogram);
      quantized_histogram.assign(quantized_histogram.size(), 0);
    }
    const auto itr_min_divergence =
        std::min_element(entropies.begin(), entropies.end());
    const auto idx_threshold =
        std::distance(entropies.begin(), itr_min_divergence);
    const int idx = idx_threshold + num_half_histogram_bins +
                    num_half_quantized_histogram_bins + 1;
    const float threshold = histogram.histogram_edge(idx);

    network_.layers(idx_layer).output_threshold(threshold);
  }

  auto& first_layer = network_.layers(0);
  first_layer.input_thresholds(std::vector<float>{input_range_.value()});

  network_.PostProcessCalibration();

  result->ranges.push_back(
      std::make_tuple("input_tensor:0", first_layer.input_thresholds(0)));

  for (auto& layer : network_.layers()) {
    const auto layer_name = layer.name();
    const auto threshold = layer.output_threshold();
    const auto layer_range = std::make_tuple(layer_name, threshold);
    result->ranges.push_back(layer_range);
  }

  return result;
}
}  // namespace calibration
}  // namespace spgraph_simulator
