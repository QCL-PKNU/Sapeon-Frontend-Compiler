#ifndef BACKENDS_DELEGATE_DUMMY_INFERENCE_DELEGATE_HPP
#define BACKENDS_DELEGATE_DUMMY_INFERENCE_DELEGATE_HPP

#include <memory>

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "backends/delegate/inference_delegate.hpp"
#include "enums/error.hpp"
#include "network/network.hpp"
#include "tl/expected.hpp"

class DummyInferenceDelegate : public InferenceDelegate {
 public:
  DummyInferenceDelegate(Backend &parent, Arguments &args);
  tl::expected<void, SimulatorError> Inference(
      std::unique_ptr<Network> &network) override;
};

#endif  // BACKENDS_DELEGATE_DUMMY_INFERENCE_DELEGATE_HPP
