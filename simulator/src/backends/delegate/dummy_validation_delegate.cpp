#include "backends/delegate/dummy_validation_delegate.hpp"

#include <memory>

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "enums/error.hpp"
#include "inference_context.hpp"
#include "network/network.hpp"
#include "tl/expected.hpp"

namespace validation {

DummyValidationDelegate::DummyValidationDelegate(Backend &parent,
                                                 Arguments &args) {}

tl::expected<void, SimulatorError> DummyValidationDelegate::Validate(
    std::unique_ptr<Network> &network) {
  DLOG(INFO) << "Skip Validate";
  return {};
};

std::shared_ptr<Tensor> DummyValidationDelegate::Inference(
    std::unique_ptr<Network> &network, const Tensor &input_tensor) {
  return nullptr;
}
}  // namespace validation
