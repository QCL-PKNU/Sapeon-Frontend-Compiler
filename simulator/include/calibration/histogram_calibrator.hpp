#ifndef CALIBRATION_HISTOGRAM_CALIBRATOR_HPP
#define CALIBRATION_HISTOGRAM_CALIBRATOR_HPP

#include <memory>
#include <vector>

#include "calibration/calibrator.hpp"
#include "calibration/histogram.hpp"
#include "calibration/max.hpp"
#include "network/network.hpp"

namespace spgraph_simulator {
namespace calibration {
class HistogramCalibrator : public Calibrator {
 public:
  HistogramCalibrator(Network& network);
  virtual ~HistogramCalibrator() {}
  void Collect(const Tensor& input_tensor) override final;
  // void LoadCheckpoint(const std::string& file_path);

 protected:
  Max input_range_;
  std::vector<Histogram> histograms_;
  Network& network_;
};
}  // namespace calibration
}  // namespace spgraph_simulator

#endif  // CALIBRATION_HISTOGRAM_CALIBRATOR_HPP
