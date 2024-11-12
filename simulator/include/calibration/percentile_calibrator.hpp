#ifndef CALIBRATION_PERCENTILE_CALIBRATOR_HPP
#define CALIBRATION_PERCENTILE_CALIBRATOR_HPP

#include <memory>
#include <optional>
#include <vector>

#include "calibration/histogram_calibrator.hpp"
#include "enums/error.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "tl/expected.hpp"

namespace spgraph_simulator {
namespace calibration {
class PercentileCalibrator : public HistogramCalibrator {
 public:
  PercentileCalibrator(Network& network, std::optional<float> percentile);
  virtual ~PercentileCalibrator() {}
  static std::unique_ptr<Calibrator> CreateInstance(
      Network& network, std::optional<float> percentile);

  std::unique_ptr<CalibrationResult> ComputeRange() override;

 private:
  std::optional<float> percentile_;
};
}  // namespace calibration
}  // namespace spgraph_simulator

#endif  // CALIBRATION_PERCENTILE_CALIBRATOR_HPP
