#include "calibration/entropy2_calibrator.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <optional>
#include <tuple>
#include <vector>

#include "calibration/histogram_calibrator.hpp"
#include "glog/logging.h"
#include "network/tensor.hpp"

namespace spgraph_simulator {
namespace calibration {
std::unique_ptr<Calibrator> Entropy2Calibrator::CreateInstance(
    Network& network, std::optional<float> percentile) {
  return std::make_unique<Entropy2Calibrator>(network);
}

Entropy2Calibrator::Entropy2Calibrator(Network& network)
    : HistogramCalibrator(network) {
  const size_t num_layers = network_.num_layers();
  histograms_.reserve(num_layers);
  for (int i = 0; i < num_layers; i++) {
    //   if (CheckSensitiveLayer(network_.layers(i))) {
    //     histograms_.push_back({80001});
    //   } else {
    //     histograms_.push_back({8001});
    histograms_.push_back({8001});
  }
}

void Entropy2Calibrator::ScaleVector(std::vector<double>& vec,
                                     const double scale) {
  for (int i = 0; i < vec.size(); ++i) {
    vec.at(i) *= scale;
  }
}

bool Entropy2Calibrator::CheckSensitiveLayer(const Layer& layer) {
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

size_t Entropy2Calibrator::GetNumQuantizedHistogramBins(
    int num_histogram_bins) {
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

double Entropy2Calibrator::CalculateScipyEntropy(
    const std::vector<double>& original_vec,
    const std::vector<double>& modified_vec) {
  double sum = 0.0;
  for (int i = 0; i < original_vec.size(); ++i) {
    if (original_vec.at(i) != 0.0) {
      sum += original_vec[i] * log(original_vec[i] / modified_vec[i]);
    }
  }
  return sum;
}

std::unique_ptr<Calibrator::CalibrationResult>
Entropy2Calibrator::ComputeRange() {
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

    std::vector<double> entropies(num_entropies);
    std::vector<size_t> quantized_histogram(num_quantized_histogram_bins);

    for (int i = num_half_quantized_histogram_bins;
         i < num_half_histogram_bins + 1; ++i) {
      const int idx_left = num_half_histogram_bins - i;
      const int idx_right = num_half_histogram_bins + i + 1;
      const int range = idx_right - idx_left;
      auto partial_histogram =
          std::vector<double>(histogram.histogram().begin() + idx_left,
                              histogram.histogram().begin() + idx_right);

      // bin0 removal
      int zero_bin_idx_temp = range / 2;
      size_t total_data =
          num_total_data -
          static_cast<size_t>(partial_histogram.at(zero_bin_idx_temp)) +
          static_cast<size_t>(partial_histogram.at(zero_bin_idx_temp + 1));
      partial_histogram.at(zero_bin_idx_temp) =
          partial_histogram.at(zero_bin_idx_temp + 1);

      // initialize clipped_partial_histogram
      auto clipped_partial_histogram = std::vector<double>(partial_histogram);
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

      // 1. space
      float step = static_cast<float>(range) /
                   static_cast<float>(num_quantized_histogram_bins);
      auto space = std::vector<float>(num_quantized_histogram_bins + 1);
      for (int j = 0; j < space.size(); ++j) {
        space.at(j) = step * static_cast<float>(j);
      }

      // 2. digited space
      auto digitized_space = std::vector<int>(range);
      for (int j = 0; j < digitized_space.size(); ++j) {
        for (int l = 0; l < space.size(); ++l) {
          // FIXME: accessing space.at(l + 1) might make out_of_range
          if (j >= space.at(l) && j < space.at(l + 1)) {
            digitized_space.at(j) = l;
            break;
          }
        }
      }

      for (int j = 0; j < digitized_space.size(); ++j) {
        if (partial_histogram.at(j) == 0) {
          digitized_space.at(j) = -1;
        }
      }

      // 3. new_density_counts
      auto new_density_counts =
          std::vector<double>(num_quantized_histogram_bins);
      auto counter = std::vector<int>(num_quantized_histogram_bins);
      for (int j = 0; j < digitized_space.size(); ++j) {
        if (digitized_space.at(j) != -1) {
          const int index = digitized_space.at(j);
          new_density_counts.at(index) += partial_histogram.at(j);
          counter.at(index)++;
        }
      }

      for (int j = 0; j < new_density_counts.size(); ++j) {
        if (counter.at(j) > 1) {  //-> if(counter[j] != 0){
          // double is temporarily used to avoid a numerical error
          // caused by a limited resolution of float type
          new_density_counts.at(j) =
              new_density_counts.at(j) / static_cast<double>(counter.at(j));
        }
      }

      // 4. new_density
      auto new_density = std::vector<double>(range);
      for (int j = 0; j < new_density.size(); ++j) {
        if (digitized_space.at(j) != -1) {
          int index = digitized_space.at(j);
          new_density.at(j) = new_density_counts.at(index);
        }
      }

      const long double sum_new_density =
          std::accumulate(new_density.begin(), new_density.end(),
                          static_cast<long double>(0.0));

      // 6. verify
      const size_t sum_outlier_count = num_left_outlier + num_right_outlier;
      const long double total_counts_new =
          sum_new_density + static_cast<double>(sum_outlier_count);

      size_t total_counts_old = 0;
      for (int j = 0; j < clipped_partial_histogram.size(); ++j) {
        total_counts_old +=
            static_cast<size_t>(clipped_partial_histogram.at(j));
      }

      DLOG_IF(FATAL,
              static_cast<size_t>(total_counts_new + 0.5) != total_data ||
                  total_counts_old != total_data)
          << "Count mismatch";

      ScaleVector(clipped_partial_histogram,
                  1.0 / static_cast<double>(total_counts_old));
      ScaleVector(new_density, 1.0 / static_cast<double>(sum_new_density));

      entropies.at(i - num_half_quantized_histogram_bins) =
          CalculateScipyEntropy(clipped_partial_histogram, new_density);
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
