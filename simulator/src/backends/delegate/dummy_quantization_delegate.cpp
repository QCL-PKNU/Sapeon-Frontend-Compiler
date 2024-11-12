#include "backends/delegate/dummy_quantization_delegate.hpp"

#define CLASS DummyQuantizationDelegate
#define SCOPE CLASS

#include <memory>
using std::unique_ptr;
#include "glog/logging.h"
#include "tl/expected.hpp"
using tl::expected;

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "enums/error.hpp"
#include "network/network.hpp"

namespace quantization {
SCOPE::DummyQuantizationDelegate(Backend &parent, Arguments &args) {}

expected<void, SimulatorError> SCOPE::Quantize(unique_ptr<Network> &network) {
  LOG(INFO) << "Skip Quantization";
  return {};
}
}  // namespace quantization
