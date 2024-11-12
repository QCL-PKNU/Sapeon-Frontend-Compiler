#ifndef ENUMS_CALIBRATION_HPP
#define ENUMS_CALIBRATION_HPP

#include <iostream>

namespace spgraph_simulator {
namespace calibration {
enum class CalibrationMethod { kMax = 0, kPercentile, kEntropy, kEntropy2 };

inline std::ostream& operator<<(std::ostream& os, CalibrationMethod method) {
  switch (method) {
    case CalibrationMethod::kMax:
      os << "max";
      break;
    case CalibrationMethod::kPercentile:
      os << "percentile";
      break;
    case CalibrationMethod::kEntropy:
      os << "entropy";
      break;
    case CalibrationMethod::kEntropy2:
      os << "entropy2";
      break;
    default:
      break;
  }
  return os;
}
}  // namespace calibration
}  // namespace spgraph_simulator

#endif  // ENUMS_CALIBRATION_HPP
