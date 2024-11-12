#include "backends/delegate/x330_inference_delegate.hpp"

#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "datatype.hpp"
#include "enums/error.hpp"
#include "factory.hpp"
#include "glog/logging.h"
#include "inference_context.hpp"
#include "network/network.hpp"
#include "tl/expected.hpp"
#include "utility.hpp"
#include "x330/aixv_utils.h"
#include "x330/x330_operation.hpp"

X330InferenceDelegate::X330InferenceDelegate(Backend &parent, Arguments &args)
    : parent_(parent), dump_(args) {
  GetAbsoluteFilePath(image_path_, args.image_path().value());
}

tl::expected<void, SimulatorError> X330InferenceDelegate::Inference(
    std::unique_ptr<Network> &network) {
  using x330::X330Operation;
  LOG(INFO) << "Inference Started\n";
  struct timespec start_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);

  auto input_tensor =
      parent_.GetInputImageTensor(image_path_, dty::DataType::FP32).value();
  InferenceContext input_ctx{*network, input_tensor};
  input_ctx.SetLayerContext({-1}, 0, 1);
  auto input_ops = Factory<x330::X330Operation>::CreateInstance("Input");
  input_ops->Forward(network->input_layer(), input_ctx);
  dump_.DumpX330NetworkInput(input_ctx.GetLayerOutputTensor(0));
  const auto converted = std::move(*input_ctx.GetLayerOutputTensor(0));

  InferenceContext ctx{*network, converted};

  for (int idx_layer = 0; idx_layer < network->num_layers(); idx_layer++) {
    auto &layer = network->layers(idx_layer);
    ctx.SetLayerContext(layer.predecessors(), idx_layer,
                        layer.operation_types().size());
    for (const auto &op_name : layer.operation_types()) {
      auto operation = Factory<x330::X330Operation>::CreateInstance(op_name);
      if (operation == nullptr) {
        DLOG(ERROR) << "Failed to create operation: " << op_name;
      }
      operation->Forward(layer, ctx);
    }
    auto activation = ctx.GetLayerOutputTensor(idx_layer);
    dump_.DumpX330LayerOutput(*activation, idx_layer);
    if (idx_layer == network->num_layers() - 1) {
      dump_.DumpNetworkOutput(activation, DumpLevel::DUMP_DEFAULT);
    }
    ctx.EraseUsedTensors();
  }

  PrintElapsedTime(start_time);
  LOG(INFO) << "Inference Finished\n";
  return {};
}
