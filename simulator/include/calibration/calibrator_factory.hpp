#ifndef CALIBRATION_CALIBRATOR_FACTORY_HPP
#define CALIBRATION_CALIBRATOR_FACTORY_HPP

#include <memory>
#include <optional>

#include "arguments.hpp"
#include "calibration/calibrator.hpp"
#include "enums/calibration.hpp"
#include "network/network.hpp"

namespace spgraph_simulator {
namespace calibration {
class CalibratorFactory {
 public:
  CalibratorFactory() {}
  std::unique_ptr<Calibrator> GetCalibrator(Network& network,
                                            CalibrationMethod method,
                                            std::optional<float> percentile);
};
}  // namespace calibration
}  // namespace spgraph_simulator

#endif  // CALIBRATION_CALIBRATOR_FACTORY_HPP
