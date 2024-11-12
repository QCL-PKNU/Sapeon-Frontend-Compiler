#ifndef BACKENDS_DELEGATE_DUMMY_CALIBRATION_DELEGATE_HPP
#define BACKENDS_DELEGATE_DUMMY_CALIBRATION_DELEGATE_HPP

#include <memory>

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "backends/delegate/calibration_delegate.hpp"
#include "enums/error.hpp"
#include "network/network.hpp"
#include "tl/expected.hpp"

class DummyCalibrationDelegate : public CalibrationDelegate {
 public:
  DummyCalibrationDelegate(Backend &parent, Arguments &args);
  tl::expected<void, SimulatorError> Calibrate(
      std::unique_ptr<Network> &network) override;
};

#endif  // BACKENDS_DELEGATE_DUMMY_CALIBRATION_DELEGATE_HPP
