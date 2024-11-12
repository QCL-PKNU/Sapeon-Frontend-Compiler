#ifndef CALIBRATION_ENTROPY2_CALIBRATOR_HPP
#define CALIBRATION_ENTROPY2_CALIBRATOR_HPP

#include <memory>
#include <optional>
#include <vector>

#include "calibration/histogram_calibrator.hpp"
#include "enums/error.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "tl/expected.hpp"

namespace spgraph_simulator {
namespace calibration {
class Entropy2Calibrator final : public HistogramCalibrator {
 public:
  Entropy2Calibrator(Network& network);
  ~Entropy2Calibrator() {}
  static std::unique_ptr<Calibrator> CreateInstance(
      Network& network, std::optional<float> percentile);

  std::unique_ptr<CalibrationResult> ComputeRange() override;

 private:
  size_t GetNumQuantizedHistogramBins(int num_histogram_bins);
  void ScaleVector(std::vector<double>& vec, double scale);
  double CalculateScipyEntropy(const std::vector<double>& original_vec,
                               const std::vector<double>& modified_vec);
  bool CheckSensitiveLayer(const Layer& layer);
};
}  // namespace calibration
}  // namespace spgraph_simulator

#endif  // CALIBRATION_ENTROPY2_CALIBRATOR_HPP
