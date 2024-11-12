#ifndef BACKENDS_DELEGATE_CALIBRATION_DELEGATE_HPP
#define BACKENDS_DELEGATE_CALIBRATION_DELEGATE_HPP

#include <memory>
#include <string>

#include "enums/error.hpp"
#include "network/network.hpp"
#include "tl/expected.hpp"

class CalibrationDelegate {
 public:
  virtual tl::expected<void, SimulatorError> Calibrate(
      std::unique_ptr<Network> &network) = 0;
  virtual ~CalibrationDelegate() {}
};

#endif  // BACKENDS_DELEGATE_CALIBRATION_DELEGATE_HPP
