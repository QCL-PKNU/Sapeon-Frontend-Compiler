#pragma once

#include <memory>
#include <string>
#include <vector>

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "backends/delegate/validation_delegate.hpp"
#include "enums/error.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "tl/expected.hpp"

namespace validation {
class FPValidationDelegate : public ValidationDelegate {
 public:
  FPValidationDelegate(Backend &parent, Arguments &args);
  tl::expected<void, SimulatorError> Validate(
      std::unique_ptr<Network> &network) override;
  std::shared_ptr<Tensor> Inference(std::unique_ptr<Network> &network,
                                    const Tensor &input_tensor) override;

 private:
  std::vector<std::string> validation_image_paths_;
  std::string validation_image_dir_;
  Backend &parent_;
};
}  // namespace validation
