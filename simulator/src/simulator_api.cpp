#include "simulator_api.hpp"

#include <filesystem>
#include <fstream>
#include <optional>

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "calibration/calibrator.hpp"
#include "calibration/calibrator_factory.hpp"
#include "enums/calibration.hpp"
#include "enums/to_underlying_type.hpp"
#include "factory.hpp"
#include "glog/logging.h"
#include "network/network.hpp"
#include "parser/parser.hpp"
#include "utility.hpp"

namespace spgraph_simulator {

std::unique_ptr<calibration::CalibratorAPI> MakeCalibrator(
    const std::string &model_path, calibration::CalibrationMethod calib_method,
    float percentile) {
  auto network = std::make_unique<Network>();
  auto parser = Factory<parser::Parser>::CreateInstance("spear_graph");
  auto result = parser->BuildNetwork(network, model_path, "cpu", false);

  auto p_calib = calibration::CalibratorFactory().GetCalibrator(
      *network, calib_method, percentile);

  return std::make_unique<calibration::CalibratorAPI>(std::move(network),
                                                      std::move(p_calib));
}

}  // namespace spgraph_simulator
