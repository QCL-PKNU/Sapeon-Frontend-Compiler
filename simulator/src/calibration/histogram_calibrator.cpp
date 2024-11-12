#include "calibration/histogram_calibrator.hpp"

#include "calibration/calibrator.hpp"
#include "glog/logging.h"
#include "inference_context.hpp"
#include "network/tensor.hpp"
#include "operations/cpu_operation.hpp"

namespace spgraph_simulator {
namespace calibration {
HistogramCalibrator::HistogramCalibrator(Network& network)
    : network_(network), input_range_{} {}

void HistogramCalibrator::Collect(const Tensor& input_tensor) {
  input_range_.Set(input_tensor);
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
    histograms_.at(idx_layer).Collect(*ctx.GetLayerOutputTensor(idx_layer));
    ctx.EraseUsedTensors();
  }
}
}  // namespace calibration
}  // namespace spgraph_simulator
