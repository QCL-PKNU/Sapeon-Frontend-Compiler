#pragma once

#include "calibration/calibrator.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace spgraph_simulator {
namespace python {

class PyCalibrator {
 public:
  PyCalibrator() = delete;
  PyCalibrator(std::unique_ptr<calibration::CalibratorAPI>&& p_calib)
      : p_calib_(std::move(p_calib)) {}

  void Collect(const pybind11::object& ndarray);
  std::unique_ptr<calibration::Calibrator::CalibrationResult> ComputeRange();

 private:
  std::unique_ptr<calibration::CalibratorAPI> p_calib_;
};

void init_calibrator(pybind11::module_ m);

}  // namespace python
}  // namespace spgraph_simulator
