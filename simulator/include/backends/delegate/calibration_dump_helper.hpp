#ifndef BACKENDS_DELEGATE_CALIBRATION_DUMP_HELPER_HPP
#define BACKENDS_DELEGATE_CALIBRATION_DUMP_HELPER_HPP

#include <memory>
#include <optional>
#include <string>

#include "arguments.hpp"
#include "calibration/calibrator.hpp"
#include "network/network.hpp"

class CalibrationDumpHelper {
 public:
  CalibrationDumpHelper(Arguments &args);
  void DumpCalibrationTable(std::unique_ptr<Network> &network);
  void DumpCalibrationTableSapeonFormat(std::unique_ptr<Network> &network);
  void DumpCalibrationTableSapeonFormat(
      std::unique_ptr<
          spgraph_simulator::calibration::Calibrator::CalibrationResult>
          &result);
  void DumpCalibratedModel(const std::string &graph_type,
                           const std::string &binary_path,
                           std::unique_ptr<Network> &network);

 private:
  bool dump_calibrated_model_;
  std::optional<std::string> calibrated_model_dump_path_;
  bool dump_calibration_table_;
  std::optional<std::string> calibration_table_dump_path_;
};

#endif  // BACKENDS_DELEGATE_CALIBRATION_DUMP_HELPER_HPP
