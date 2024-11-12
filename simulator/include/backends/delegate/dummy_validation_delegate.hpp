#pragma once

#include <memory>

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "backends/delegate/validation_delegate.hpp"
#include "enums/error.hpp"
#include "inference_context.hpp"
#include "network/network.hpp"
#include "tl/expected.hpp"

namespace validation {
class DummyValidationDelegate : public ValidationDelegate {
 public:
  DummyValidationDelegate(Backend &parent, Arguments &args);
  tl::expected<void, SimulatorError> Validate(
      std::unique_ptr<Network> &network) override;
  std::shared_ptr<Tensor> Inference(std::unique_ptr<Network> &network,
                                    const Tensor &input_tensor) override;
};
}  // namespace validation
