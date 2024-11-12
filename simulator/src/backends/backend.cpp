#include "backends/backend.hpp"

#define CLASS Backend
#define SCOPE CLASS

#include <cassert>
#include <memory>
using std::make_unique;
using std::shared_ptr;
using std::unique_ptr;
#include <string>
using std::string;
#include <vector>

#include "glog/logging.h"
#include "npy.hpp"
#include "tl/expected.hpp"
using tl::expected;
using tl::make_unexpected;
using tl::unexpected;

#include "backends/backend_input_helper.hpp"
#include "backends/delegate/delegate_factory.hpp"
#include "datatype.hpp"
#include "enums/error.hpp"
#include "factory.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "parser/parser.hpp"
#include "utility.hpp"

expected<void, SimulatorError> SCOPE::Run(Arguments &args) {
  auto init_backend_result = InitBackend(args);
  if (!init_backend_result) {
    DLOG(ERROR) << "InitBackend failed";
    return make_unexpected(init_backend_result.error());
  }
  DLOG(INFO) << "InitBackend done";

  auto collect_result = Collect();
  if (!collect_result) {
    DLOG(ERROR) << "Collect failed";
    return make_unexpected(collect_result.error());
  }
  DLOG(INFO) << "Collect done";

  auto calibrate_result = Calibrate();
  if (!calibrate_result) {
    DLOG(ERROR) << "Calibrate failed";
    return make_unexpected(calibrate_result.error());
  }
  DLOG(INFO) << "Calibrate done";

  auto quantize_result = Quantize();
  if (!quantize_result) {
    DLOG(ERROR) << "Quantize failed";
    return make_unexpected(quantize_result.error());
  }
  DLOG(INFO) << "Quantize done";

  auto inference_result = Inference();
  if (!inference_result) {
    DLOG(ERROR) << "Inference failed";
    return make_unexpected(inference_result.error());
  }
  DLOG(INFO) << "Inference done";

  auto validation_result = Validate();
  if (!validation_result) return make_unexpected(validation_result.error());

  return {};
}

expected<void, SimulatorError> SCOPE::InitBackend(Arguments &args) {
  LOG(INFO) << "Backend Initialization Started\n";
  // Parse Graph Binary to Network
  network_ = make_unique<Network>();
  auto parser = Factory<parser::Parser>::CreateInstance(args.graph_type());
  if (parser == nullptr) {
    const string msg = "Failed to create parser: " + args.graph_type();
    LOG(ERROR) << msg;
    return unexpected<SimulatorError>(SimulatorError::kCreateInstanceError);
  }
  auto result = parser->BuildNetwork(network_, args.model_path(),
                                     args.backend(), args.do_quant());
  if (!result) {
    DLOG(ERROR) << "BuildNetwork failed";
    return make_unexpected(result.error());
  }

  // Get Arguments
  args_ = std::move(args);

  auto factory = DelegateFactory();
  calib_ = factory.GetCalibrationDelegate(*this, args_);
  collect_ = factory.GetCollectDelegate(*this, args_);
  quant_ = factory.GetQuantizationDelegate(*this, args_);
  infer_ = factory.GetInferenceDelegate(*this, args_);
  valid_ = factory.GetValidationDelegate(*this, args_);

  // Initialize Input Helper
  input_helper_ = make_unique<BackendInputHelper>(
      args_, network_->layers(0).input_dimensions(0));
  LOG(INFO) << "Backend Initialization Finished\n";

  // TODO: add error handling logic
  return {};
}

expected<void, SimulatorError> SCOPE::Calibrate() {
  return calib_->Calibrate(network_);
}

expected<void, SimulatorError> SCOPE::Collect() {
  return collect_->Collect(network_);
}

expected<void, SimulatorError> SCOPE::Quantize() {
  return quant_->Quantize(network_);
}

expected<void, SimulatorError> SCOPE::Inference() {
  return infer_->Inference(network_);
}

expected<void, SimulatorError> SCOPE::Validate() {
  return valid_->Validate(network_);
}

expected<Tensor, SimulatorError> SCOPE::GetInputNumpyTensor(
    const string &input_file_path) {
  std::vector<npy::ndarray_len_t> shape;
  std::vector<float> data;

  try {
    npy::LoadArrayFromNumpy(input_file_path, shape, data);
  } catch (const std::runtime_error &e) {
    LOG(ERROR) << e.what() << '\n';
    return make_unexpected(SimulatorError::kNumpyLoadError);
  }
  // assumes that the input tensor shape is [c,h,w]
  if (shape.size() != 3) {
    LOG(ERROR) << "Invalid Tensor Shape, ndims = " << shape.size() << '\n';
    return make_unexpected(SimulatorError::kTensorShapeError);
  }
  const auto c = shape[0], h = shape[1], w = shape[2];
  const auto &model_input_dim = network_->layers(0).input_dimensions(0);
  if (c != model_input_dim.c() || h != model_input_dim.h() ||
      w != model_input_dim.w()) {
    LOG(ERROR) << "Invalid Tensor Shape: " << c << ":" << model_input_dim.c()
               << "," << h << ":" << model_input_dim.h() << "," << w << ":"
               << model_input_dim.w() << "\n";
    return make_unexpected(SimulatorError::kTensorShapeError);
  }

  Tensor tensor{1, c, h, w, dty::DataType::FP32};

  float *tensor_ptr = tensor.data<float>();
  std::memcpy(tensor_ptr, data.data(), sizeof(float) * c * h * w);

  return tensor;
}

expected<Tensor, SimulatorError> SCOPE::GetInputImageTensor(
    const string &input_file_path, dty::DataType dtype) {
  switch (dtype) {
    case dty::DataType::FP32:
      return input_helper_->GetInputImageTensor(input_file_path);
    case dty::DataType::SINT8:
      return input_helper_->GetInputImageTensor(
          input_file_path, network_->layers(0).input_thresholds(0));
    default:
      const string msg = "Unknown Data Type: " + dty::NameOf(dtype);
      LOG(ERROR) << msg;
      return tl::make_unexpected(SimulatorError::kInvalidDataType);
  }
}

tl::expected<Tensor, SimulatorError> SCOPE::FuseInputTensors(
    const std::vector<Tensor> &tensors) {
  return input_helper_->FuseInputTensors(tensors);
}
