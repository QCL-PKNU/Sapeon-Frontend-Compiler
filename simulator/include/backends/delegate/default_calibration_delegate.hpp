#ifndef BACKENDS_DELEGATE_DEFAULT_CALIBRATION_DELEGATE_HPP
#define BACKENDS_DELEGATE_DEFAULT_CALIBRATION_DELEGATE_HPP

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "backends/delegate/calibration_delegate.hpp"
#include "backends/delegate/calibration_dump_helper.hpp"
#include "enums/calibration.hpp"
#include "enums/error.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "tl/expected.hpp"

class DefaultCalibrationDelegate : public CalibrationDelegate {
 public:
  DefaultCalibrationDelegate(Backend& parent, Arguments& args);
  tl::expected<void, SimulatorError> Calibrate(
      std::unique_ptr<Network>& network) override;
  virtual ~DefaultCalibrationDelegate() {}

 private:
  tl::expected<Tensor, SimulatorError> GetInputTensor(
      Arguments::InputType input_type, const std::string& input_file_path);
  Arguments::InputType input_type_;
  CalibrationDumpHelper dump_;
  Backend& parent_;
  std::string graph_type_;
  std::string graph_binary_path_;
  std::string backend_type_;
  spgraph_simulator::calibration::CalibrationMethod calibration_mode_;
  size_t batch_size_;
  std::optional<float> percentile_;
  std::vector<std::string> calibration_image_paths_;
  std::vector<std::shared_ptr<Tensor>> activations_;
  std::shared_ptr<Tensor> activation_;
};

#endif  // BACKENDS_DELEGATE_DEFAULT_CALIBRATION_DELEGATE_HPP
