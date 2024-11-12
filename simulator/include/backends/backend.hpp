#ifndef BACKENDS_BACKEND_HPP
#define BACKENDS_BACKEND_HPP

#include <memory>
#include <string>

#include "arguments.hpp"
#include "backends/backend_input_helper.hpp"
#include "backends/delegate/calibration_delegate.hpp"
#include "backends/delegate/collect_delegate.hpp"
#include "backends/delegate/inference_delegate.hpp"
#include "backends/delegate/quantization_delegate.hpp"
#include "backends/delegate/validation_delegate.hpp"
#include "datatype.hpp"
#include "enums/error.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "tl/expected.hpp"

class Backend {
 public:
  tl::expected<void, SimulatorError> Run(Arguments &args);
  tl::expected<void, SimulatorError> InitBackend(Arguments &args);
  tl::expected<Tensor, SimulatorError> GetInputImageTensor(
      const std::string &input_file_path, dty::DataType dtype);
  tl::expected<Tensor, SimulatorError> GetInputNumpyTensor(
      const std::string &input_file_path);
  tl::expected<Tensor, SimulatorError> FuseInputTensors(
      const std::vector<Tensor> &tensors);
  virtual ~Backend() {}

 protected:
  tl::expected<void, SimulatorError> Calibrate();
  tl::expected<void, SimulatorError> Collect();
  tl::expected<void, SimulatorError> Quantize();
  tl::expected<void, SimulatorError> Inference();
  tl::expected<void, SimulatorError> Validate();
  std::unique_ptr<BackendInputHelper> input_helper_;
  std::unique_ptr<InferenceDelegate> infer_;
  std::unique_ptr<CalibrationDelegate> calib_;
  std::unique_ptr<collect::CollectDelegate> collect_;
  std::unique_ptr<quantization::QuantizationDelegate> quant_;
  std::unique_ptr<validation::ValidationDelegate> valid_;
  Arguments args_;
  std::unique_ptr<Network> network_;
};

#endif  // BACKENDS_BACKEND_HPP
