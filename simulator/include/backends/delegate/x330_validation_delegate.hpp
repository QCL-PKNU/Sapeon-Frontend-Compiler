#pragma once

#include <memory>
#include <string>
#include <vector>

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "backends/delegate/classification_validation_helper.hpp"
#include "backends/delegate/validation_delegate.hpp"
#include "enums/error.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "tl/expected.hpp"

namespace validation {
class X330ValidationDelegate : public ValidationDelegate {
 public:
  X330ValidationDelegate(Backend &parent, Arguments &args);
  tl::expected<void, SimulatorError> Validate(
      std::unique_ptr<Network> &network) override;
  std::shared_ptr<Tensor> Inference(std::unique_ptr<Network> &network,
                                    const Tensor &input_tensor) override;

 private:
  std::string validation_image_dir_;
  std::vector<std::string> validation_image_paths_;
  Backend &parent_;
};
}  // namespace validation
