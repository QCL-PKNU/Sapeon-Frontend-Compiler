#ifndef CALIBRATION_CALIBRATOR_HPP
#define CALIBRATION_CALIBRATOR_HPP

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "network/network.hpp"
#include "network/tensor.hpp"

namespace spgraph_simulator {
namespace calibration {
class Calibrator {
 public:
  virtual ~Calibrator() {}
  class CalibrationResult {
   public:
    void AsTextfile(const std::string& path_to_out_textfile);
    std::vector<std::tuple<std::string, float>> ranges;
  };

  virtual void Collect(const Tensor& input_tensor) = 0;
  virtual std::unique_ptr<CalibrationResult> ComputeRange() = 0;
};

class CalibratorAPI {
 public:
  CalibratorAPI() = delete;
  CalibratorAPI(std::unique_ptr<Network>&& network,
                std::unique_ptr<Calibrator>&& calibrator)
      : network_(std::move(network)), calibrator_(std::move(calibrator)) {}

  void Collect(const Tensor& input_tensor) {
    calibrator_->Collect(input_tensor);
  }
  std::unique_ptr<Calibrator::CalibrationResult> ComputeRange() {
    return calibrator_->ComputeRange();
  }

 private:
  std::unique_ptr<Network> network_;
  std::unique_ptr<Calibrator> calibrator_;
};
}  // namespace calibration
}  // namespace spgraph_simulator

#endif  // CALIBRATION_CALIBRATOR_HPP
