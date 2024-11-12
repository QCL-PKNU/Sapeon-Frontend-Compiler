#include "backends/delegate/dummy_inference_delegate.hpp"

#define SCOPE DummyInferenceDelegate

#include <memory>
using std::unique_ptr;

#include "tl/expected.hpp"
using tl::expected;

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "enums/error.hpp"
#include "network/network.hpp"

SCOPE::DummyInferenceDelegate(Backend &parent, Arguments &args) {}

expected<void, SimulatorError> SCOPE::Inference(unique_ptr<Network> &network) {
  LOG(INFO) << "Skip Inference";
  return {};
}
