#ifndef CALIBRATION_MAX_CALIBRATOR_HPP
#define CALIBRATION_MAX_CALIBRATOR_HPP

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "calibration/calibrator.hpp"
#include "calibration/max.hpp"
#include "enums/error.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "tl/expected.hpp"

namespace spgraph_simulator {
namespace calibration {
class MaxCalibrator final : public Calibrator {
 public:
  MaxCalibrator(Network& network);
  ~MaxCalibrator() {}
  static std::unique_ptr<Calibrator> CreateInstance(
      Network& network, std::optional<float> percentile);

  void Collect(const Tensor& input_tensor) override;
  std::unique_ptr<CalibrationResult> ComputeRange() override;

 private:
  std::vector<Max> maxes_;
  Network& network_;
};
}  // namespace calibration
}  // namespace spgraph_simulator

#endif  // CALIBRATION_MAX_CALIBRATOR_HPP
