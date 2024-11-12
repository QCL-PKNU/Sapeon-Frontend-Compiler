#include "calibration/max_calibrator.hpp"

#include <algorithm>
#include <memory>
#include <tuple>
#include <vector>

#include "calibration/calibrator.hpp"
#include "calibration/max.hpp"
#include "inference_context.hpp"
#include "network/tensor.hpp"
#include "operations/cpu_operation.hpp"

namespace spgraph_simulator {
namespace calibration {
std::unique_ptr<Calibrator> MaxCalibrator::CreateInstance(
    Network& network, std::optional<float> percentile) {
  return std::make_unique<MaxCalibrator>(network);
}

MaxCalibrator::MaxCalibrator(Network& network) : network_(network) {
  maxes_ = std::vector<Max>(network_.num_layers() + 1);
}

void MaxCalibrator::Collect(const Tensor& input_tensor) {
  maxes_.at(0).Set(input_tensor);
  InferenceContext ctx{network_, input_tensor};

  for (int idx_layer = 0; idx_layer < network_.num_layers(); idx_layer++) {
    auto& layer = network_.layers(idx_layer);
    ctx.SetLayerContext(layer.predecessors(), idx_layer,
                        layer.operation_types().size());
    for (const auto& op_name : layer.operation_types()) {
      auto operation = Factory<CpuOperation>::CreateInstance(op_name);
      if (operation == nullptr) {
        DLOG(ERROR) << "Failed to create operation: " << op_name;
      }
      operation->Forward(layer, ctx);
    }
    auto max_result =
        maxes_.at(idx_layer + 1).Set(*ctx.GetLayerOutputTensor(idx_layer));
    // if (!max_result.has_value()) {
    //   return tl::make_unexpected(max_result.error());
    // }
    ctx.EraseUsedTensors();
  }
}

std::unique_ptr<Calibrator::CalibrationResult> MaxCalibrator::ComputeRange() {
  const size_t num_layers = network_.num_layers();
  auto result = std::make_unique<Calibrator::CalibrationResult>();
  result->ranges.reserve(num_layers);

  auto& first_layer = network_.layers(0);
  first_layer.input_thresholds(std::vector<float>{maxes_.at(0).value()});

  for (int idx_layer = 0; idx_layer < num_layers; idx_layer++) {
    auto threshold = maxes_.at(idx_layer + 1).value();
    network_.layers(idx_layer).output_threshold(threshold);
  }

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
