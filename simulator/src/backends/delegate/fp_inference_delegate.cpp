#include "backends/delegate/fp_inference_delegate.hpp"

#define SCOPE FPInferenceDelegate

#include <memory>

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "datatype.hpp"
#include "enums/error.hpp"
#include "factory.hpp"
#include "glog/logging.h"
#include "inference_context.hpp"
#include "network/network.hpp"
#include "operations/cpu_operation.hpp"
#include "tl/expected.hpp"
#include "utility.hpp"

SCOPE::FPInferenceDelegate(Backend &parent, Arguments &args)
    : parent_(parent), dump_(args) {
  GetAbsoluteFilePath(image_path_, args.image_path().value());
}

tl::expected<void, SimulatorError> SCOPE::Inference(
    std::unique_ptr<Network> &network) {
  LOG(INFO) << "Inference Started\n";
  struct timespec start_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);

  auto image_tensor =
      parent_.GetInputImageTensor(image_path_, dty::DataType::FP32).value();
  dump_.DumpNetworkInput(std::make_shared<Tensor>(image_tensor),
                         DumpLevel::DUMP_DEFAULT);

  InferenceContext ctx{*network, image_tensor};

  for (int idx_layer = 0; idx_layer < network->num_layers(); ++idx_layer) {
    Layer &layer = network->layers(idx_layer);
    ctx.SetLayerContext(layer.predecessors(), idx_layer,
                        layer.operation_types().size());

    for (const auto &op_name : layer.operation_types()) {
      auto operation = Factory<CpuOperation>::CreateInstance(op_name);
      if (operation == nullptr) {
        DLOG(ERROR) << "Failed to create operation: " << op_name;
      }
      operation->Forward(layer, ctx);
    }
    auto activation = ctx.GetLayerOutputTensor(idx_layer);
    dump_.DumpLayerOutput(activation, idx_layer, DumpLevel::DUMP_DEBUG);
    if (idx_layer == network->num_layers() - 1) {
      dump_.DumpNetworkOutput(activation, DumpLevel::DUMP_DEFAULT);
    }
    ctx.EraseUsedTensors();
  }

  PrintElapsedTime(start_time);
  LOG(INFO) << "Inference Finished\n";
  return {};
}
