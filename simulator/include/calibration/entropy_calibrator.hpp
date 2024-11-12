#ifndef CALIBRATION_ENTROPY_CALIBRATOR_HPP
#define CALIBRATION_ENTROPY_CALIBRATOR_HPP

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
class EntropyCalibrator final : public HistogramCalibrator {
 public:
  EntropyCalibrator(Network& network);
  ~EntropyCalibrator() {}
  static std::unique_ptr<Calibrator> CreateInstance(
      Network& network, std::optional<float> percentile);

  std::unique_ptr<CalibrationResult> ComputeRange() override;

 private:
  size_t GetNumQuantizedHistogramBins(int num_histogram_bins);
  tl::expected<void, SimulatorError> SmoothDistribution(std::vector<float>& vec,
                                                        float epsilon);
  float CalculateEntropy(const std::vector<float>& original_vec,
                         const std::vector<float>& quantized_vec);
  bool CheckSensitiveLayer(const Layer& layer);
};
}  // namespace calibration
}  // namespace spgraph_simulator

#endif  // CALIBRATION_ENTROPY_CALIBRATOR_HPP
