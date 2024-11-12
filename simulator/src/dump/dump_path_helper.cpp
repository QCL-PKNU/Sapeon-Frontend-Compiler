#include "dump/dump_path_helper.hpp"

#include <iomanip>
#include <memory>
using std::make_unique;
using std::unique_ptr;
#include <sstream>
using std::stringstream;
#include <string>
using std::string;

#include "arguments.hpp"
#include "enums/dump.hpp"
#include "glog/logging.h"

DumpPathHelper::DumpPathHelper(Arguments &args) {
  dump_level_ = GetDumpLevel(args.dump_level());
  dump_dir_ = args.dump_dir();
}

string DumpPathHelper::GetLayerOutputPath(int idx_layer) {
  if (dump_level_ == DumpLevel::DUMP_NONE) {
    LOG(INFO) << "dump-level is none. Does not need LayerOutputPath";
    return "";
  }
  stringstream str_idx_layer;
  str_idx_layer << std::setw(3) << std::setfill('0') << idx_layer;

  string output_file_path =
      dump_dir_.value() + "/layer" + str_idx_layer.str() + ".output";
  return output_file_path;
}

string DumpPathHelper::GetActivationIntOutputPath(int idx_layer) {
  if (dump_level_ == DumpLevel::DUMP_NONE) {
    LOG(INFO) << "dump-level is none. Does not need ActivationIntOutputPath";
    return "";
  }
  stringstream str_idx_layer;
  str_idx_layer << std::setw(3) << std::setfill('0') << idx_layer;

  string output_file_path =
      dump_dir_.value() + "/act_int_" + str_idx_layer.str() + ".txt";
  return output_file_path;
}

string DumpPathHelper::GetActivationFloatOutputPath(int idx_layer) {
  if (dump_level_ == DumpLevel::DUMP_NONE) {
    LOG(INFO) << "dump-level is none. Does not need ActivationFloatOutputPath";
    return "";
  }
  stringstream str_idx_layer;
  str_idx_layer << std::setw(3) << std::setfill('0') << idx_layer;

  string output_file_path =
      dump_dir_.value() + "/act_fp_" + str_idx_layer.str() + ".txt";
  return output_file_path;
}

string DumpPathHelper::GetFilePath(const string &file_path) {
  if (dump_level_ == DumpLevel::DUMP_NONE) {
    LOG(INFO) << "dump-level is none. Does not need FilePath";
    return "";
  }
  return dump_dir_.value_or("dump") + "/" + file_path;
}
