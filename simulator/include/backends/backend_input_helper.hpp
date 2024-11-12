#ifndef BACKENDS_BACKEND_INPUT_HELPER_HPP
#define BACKENDS_BACKEND_INPUT_HELPER_HPP

#include <memory>
#include <optional>
#include <string>

#include "arguments.hpp"
#include "enums/error.hpp"
#include "network/dimension.hpp"
#include "network/tensor.hpp"
#include "tl/expected.hpp"

class BackendInputHelper {
 public:
  BackendInputHelper(Arguments& args, Dimension input_dimension);
  Tensor GetInputImageTensor(const std::string& input_file_path,
                             float threshold);
  Tensor GetInputImageTensor(const std::string& input_file_path);
  float GetNetworkInputThreshold();
  tl::expected<Tensor, SimulatorError> FuseInputTensors(
      const std::vector<Tensor>& tensors);

 private:
  std::optional<std::string> preprocess_config_file_path_;
  Dimension dimension_;
};

#endif  // BACKENDS_BACKEND_INPUT_HELPER_HPP
