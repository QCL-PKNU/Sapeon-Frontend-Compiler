#include "calibration/calibrator_factory.hpp"

#include <memory>
#include <optional>

#include "calibration/calibrator.hpp"
#include "calibration/entropy2_calibrator.hpp"
#include "calibration/entropy_calibrator.hpp"
#include "calibration/max_calibrator.hpp"
#include "calibration/percentile_calibrator.hpp"
#include "enums/to_underlying_type.hpp"
#include "glog/logging.h"
#include "network/network.hpp"

namespace spgraph_simulator {
namespace calibration {
std::unique_ptr<calibration::Calibrator> CalibratorFactory::GetCalibrator(
    Network& network, CalibrationMethod method,
    std::optional<float> percentile) {
  switch (method) {
    case CalibrationMethod::kMax:
      return MaxCalibrator::CreateInstance(network, percentile);
    case CalibrationMethod::kPercentile:
      return PercentileCalibrator::CreateInstance(network, percentile);
    case CalibrationMethod::kEntropy:
      return EntropyCalibrator::CreateInstance(network, percentile);
    case CalibrationMethod::kEntropy2:
      return Entropy2Calibrator::CreateInstance(network, percentile);
    default:
      LOG(ERROR) << "unexpected CalibrationMethod: "
                 << spgraph_simulator::ToUnderlyingType(method);
      exit(1);
  }
}
}  // namespace calibration
}  // namespace spgraph_simulator
