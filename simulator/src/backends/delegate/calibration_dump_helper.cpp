#include "backends/delegate/calibration_dump_helper.hpp"

#define CLASS CalibrationDumpHelper
#define SCOPE CLASS

#include <memory>
using std::unique_ptr;
#include <string>
using std::string;
#include "arguments.hpp"
#include "factory.hpp"
#include "glog/logging.h"
#include "network/network.hpp"
#include "parser/parser.hpp"

SCOPE::CalibrationDumpHelper(Arguments &args) {
  dump_calibration_table_ = args.dump_calibration_table();
  calibration_table_dump_path_ = args.calibration_table_dump_path();
  dump_calibrated_model_ = args.dump_calibrated_model();
  calibrated_model_dump_path_ = args.calibrated_model_dump_path();
}

void SCOPE::DumpCalibrationTable(unique_ptr<Network> &network) {
  if (!dump_calibration_table_) {
    return;
  }
  FILE *fp = fopen(calibration_table_dump_path_.value().c_str(), "w");

  auto in_thr = network->layers(0).input_thresholds(0);
  fprintf(fp, "%3d     %14s  input   %8.6f    %8.6f\n", 0, "Input", in_thr,
          127.f / in_thr);

  for (int i = 0; i < network->num_layers(); ++i) {
    float thr = network->layers(i).output_threshold();
    fprintf(fp, "%3d     %14s output   %8.6f    %8.6f\n", i,
            network->layers(i).operation_types(0).c_str(), thr, 127.f / thr);
  }
  fclose(fp);
}

void SCOPE::DumpCalibrationTableSapeonFormat(unique_ptr<Network> &network) {
  if (!dump_calibration_table_) {
    return;
  }
  FILE *fp = fopen(calibration_table_dump_path_.value().c_str(), "w");

  auto in_thr = network->layers(0).input_thresholds(0);
  fprintf(fp, "input_tensor:0\t%8.6f\n", in_thr);

  for (int i = 0; i < network->num_layers(); ++i) {
    float thr = network->layers(i).output_threshold();
    fprintf(fp, "%s\t%8.6f\n", network->layers(i).name().c_str(), thr);
  }
  fclose(fp);
}

void SCOPE::DumpCalibrationTableSapeonFormat(
    unique_ptr<spgraph_simulator::calibration::Calibrator::CalibrationResult>
        &result) {
  if (!dump_calibration_table_) {
    return;
  }
  result->AsTextfile(calibration_table_dump_path_.value());
}

void SCOPE::DumpCalibratedModel(const string &graph_type,
                                const string &binary_path,
                                unique_ptr<Network> &network) {
  if (!dump_calibrated_model_) {
    return;
  }
  auto parser = Factory<parser::Parser>::CreateInstance(graph_type);
  parser->DumpCalibratedModel(network, binary_path,
                              calibrated_model_dump_path_.value());
}
