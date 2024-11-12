#ifndef SIMULATOR_API_HPP
#define SIMULATOR_API_HPP

#include <cstdint>
#include <string>

#include "calibration/calibrator.hpp"
#include "enums/calibration.hpp"

namespace spgraph_simulator {

/**
 * model_path: path to spgraph binary
 * calib_method: calibration algorithm to use
 * percentile: Percentile calibration config
 */
std::unique_ptr<calibration::CalibratorAPI> MakeCalibrator(
    const std::string &model_path, calibration::CalibrationMethod calib_method,
    float percentile);

}  // namespace spgraph_simulator

#endif  // SIMULATOR_API_HPP
