#include "calibration/percentile_calibrator.hpp"

#include <memory>
#include <numeric>
#include <optional>
#include <tuple>
#include <vector>

#include "calibration/histogram_calibrator.hpp"
#include "enums/error.hpp"
#include "network/tensor.hpp"

namespace spgraph_simulator {
namespace calibration {
std::unique_ptr<Calibrator> PercentileCalibrator::CreateInstance(
    Network& network, std::optional<float> percentile) {
  return std::make_unique<PercentileCalibrator>(network, percentile);
}

PercentileCalibrator::PercentileCalibrator(Network& network,
                                           std::optional<float> percentile)
    : HistogramCalibrator(network), percentile_(percentile) {
  const size_t num_layers = network_.num_layers();
  histograms_.reserve(num_layers);
  for (int i = 0; i < num_layers; i++) {
    histograms_.push_back({80001});
  }
}

std::unique_ptr<Calibrator::CalibrationResult>
PercentileCalibrator::ComputeRange() {
  const size_t num_layers = network_.num_layers();
  auto result = std::make_unique<Calibrator::CalibrationResult>();
  result->ranges.reserve(num_layers + 1);

  if (!percentile_.has_value()) {
    LOG(ERROR) << "Percentile is not provided, do max calibration";
    percentile_ = 1.0;
  }
  const float target_ratio = 1 - percentile_.value();

  for (int idx_layer = 0; idx_layer < num_layers; idx_layer++) {
    const auto& histogram = histograms_.at(idx_layer);
    const size_t num_total_data = histogram.TotalCount();

    const size_t num_histogram_bins = histogram.num_bins();
    const size_t idx_middle = num_histogram_bins / 2;

    int i;
    size_t num_left_outliers = 0;
    size_t num_right_outliers = 0;
    for (i = idx_middle; i >= 0; --i) {
      const int idx_left = idx_middle - i;
      const int idx_right = idx_middle + i;

      num_left_outliers += histogram.histogram(idx_left);
      num_right_outliers += histogram.histogram(idx_right);

      const float cur_saturation_ratio =
          static_cast<float>(num_left_outliers + num_right_outliers) /
          static_cast<float>(num_total_data);
      if (cur_saturation_ratio >= target_ratio) break;
    }
    float threshold = histogram.histogram_edge(idx_middle + i + 1);

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
