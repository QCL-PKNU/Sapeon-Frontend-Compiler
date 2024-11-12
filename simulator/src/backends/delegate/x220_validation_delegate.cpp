#include "backends/delegate/x220_validation_delegate.hpp"

#include <memory>

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "backends/delegate/classification_validation_helper.hpp"
#include "datatype.hpp"
#include "enums/error.hpp"
#include "glog/logging.h"
#include "inference_context.hpp"
#include "network/network.hpp"
#include "operations/cpu_operation.hpp"
#include "tl/expected.hpp"
#include "utility.hpp"

namespace validation {

X220ValidationDelegate::X220ValidationDelegate(Backend &parent, Arguments &args)
    : parent_(parent) {
  // Parse Arguments
  validation_image_dir_ = args.validation_image_dir().value();
  GetImageFilePaths(validation_image_paths_, validation_image_dir_);
}

tl::expected<void, SimulatorError> X220ValidationDelegate::Validate(
    std::unique_ptr<Network> &network) {
  LOG(INFO) << "Validate Started\n";
  struct timespec start_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);

  auto valid = validation::ClassificationValidationHelper();

  auto class_dirs = valid.GetClassDirs(validation_image_dir_);

  if (class_dirs.size() == 0) {
    LOG(ERROR)
        << "The structure of the validation image directory is incorrect.";
    return tl::make_unexpected(SimulatorError::kValidationError);
  }

  size_t num_correct = 0;
  size_t num_count = 0;
  for (const auto &image_path : validation_image_paths_) {
    auto input_tensor =
        parent_.GetInputImageTensor(image_path, dty::DataType::SINT8).value();
    auto output_activation = Inference(network, input_tensor);

    auto validation_result =
        valid.ValidateTopOneIndex(class_dirs, image_path, output_activation);
    if (!validation_result.has_value()) {
      return tl::make_unexpected(validation_result.error());
    }

    if (validation_result.value()) {
      num_correct++;
    }
    num_count++;

    // LOG
    double top_1_percent =
        (static_cast<double>(num_correct) / static_cast<double>(num_count)) *
        100;
    LOG(INFO) << num_count << "/" << validation_image_paths_.size() << ", "
              << top_1_percent << "%";
  }

  PrintElapsedTime(start_time);
  LOG(INFO) << "Validate Finished\n";
  return {};
}

std::shared_ptr<Tensor> X220ValidationDelegate::Inference(
    std::unique_ptr<Network> &network, const Tensor &input_tensor) {
  InferenceContext ctx{*network, input_tensor};

  for (int i = 0; i < network->num_layers(); i++) {
    auto &layer = network->layers(i);
    ctx.SetLayerContext(layer.predecessors(), i, layer.operation_types().size(),
                        layer.x220_quant_config().out_dtype());
    for (const auto &op_name : layer.operation_types()) {
      auto operation = Factory<CpuOperation>::CreateInstance(op_name);
      if (operation == nullptr) {
        DLOG(ERROR) << "Failed to create operation: " << op_name;
      }
      operation->Forward(layer, ctx);
    }
  }
  return ctx.GetLayerOutputTensor(network->num_layers() - 1);
}
}  // namespace validation
