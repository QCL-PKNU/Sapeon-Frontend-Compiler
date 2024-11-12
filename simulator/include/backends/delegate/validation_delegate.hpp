#pragma once

#include <memory>
#include <string>

#include "enums/error.hpp"
#include "network/network.hpp"
#include "tl/expected.hpp"

namespace validation {
class ValidationDelegate {
 public:
  virtual tl::expected<void, SimulatorError> Validate(
      std::unique_ptr<Network> &network) = 0;
  virtual std::shared_ptr<Tensor> Inference(std::unique_ptr<Network> &network,
                                            const Tensor &input_tensor) = 0;
  virtual ~ValidationDelegate() {}
};
}  // namespace validation
