#include "backends/delegate/dummy_calibration_delegate.hpp"

#define SCOPE DummyCalibrationDelegate

#include <memory>
using std::unique_ptr;

#include "tl/expected.hpp"
using tl::expected;

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "enums/error.hpp"
#include "network/network.hpp"

SCOPE::DummyCalibrationDelegate(Backend &parent, Arguments &args) {}

expected<void, SimulatorError> SCOPE::Calibrate(unique_ptr<Network> &network) {
  LOG(INFO) << "Skip Calibration";
  return {};
}
