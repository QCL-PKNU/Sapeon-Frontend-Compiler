 #include "arguments.hpp"

#define SCOPE Arguments

#include <algorithm>
#include <optional>
using std::optional;
#include <string>
using std::string;
#include <vector>
using std::vector;

#include "glog/logging.h"
#include "tl/expected.hpp"
using tl::expected;
using tl::make_unexpected;

#include "enums/calibration.hpp"
#include "enums/error.hpp"
#include "utility.hpp"

expected<void, SimulatorError> SCOPE::CheckArguments() {
  expected<void, SimulatorError> result;

  if (!do_calib_ && !do_collect_ && !do_quant_ && !do_infer_ && !do_valid_) {
    LOG(ERROR) << "at least one of --calib, --collect, --quant, --infer, or "
                  "--valid is required";
    return make_unexpected(SimulatorError::kArgumentsParsingError);
  }

  // backend
  const vector<string> backend_options{"cudnn", "cpu"};
  result = CheckArgInList("backend", backend_, backend_options);
  if (!result) return make_unexpected(result.error());

  // graph-type
  const vector<string> graph_type_options{"spear_graph", "aix_graph"};
  result = CheckArgInList("--graph-type", graph_type_, graph_type_options);
  if (!result) return make_unexpected(result.error());

  // model-path
  result = CheckStringFilePathReadable("--model-path", model_path_);
  if (!result) return make_unexpected(result.error());

  // preprocess-config-path
  if (preprocess_config_path_.has_value()) {
    result = CheckStringFilePathReadable("--preprocess-config-path",
                                         preprocess_config_path_.value());
    if (!result) return make_unexpected(result.error());
  }

  // dump-level
  const vector<string> dump_level_options{"none", "default", "debug"};
  result = CheckArgInList("--dump-level", dump_level_, dump_level_options);
  if (!result) return make_unexpected(result.error());

  // dump-dir
  if (dump_level_ != "none") {
    result = CheckArgExist("--dump-dir", dump_dir_,
                           "required from --dump-level=" + dump_level_);
    if (!result) return make_unexpected(result.error());
    result = CheckStringDirectory("--dump-dir", dump_dir_.value());
    if (!result) return make_unexpected(result.error());
  } else {
    result = CheckArgNotExist("--dump-dir", dump_dir_,
                              "required from --dump-level=none");
    if (!result) return make_unexpected(result.error());
  }

  // calib related arguments
  if (do_calib_) {
    result = CheckCalibArguments();
    if (!result) return make_unexpected(result.error());
  } else {
    const string error_msg = " is ignored: --calib=false";
    if (calibration_method_.has_value()) {
      LOG(WARNING) << "--calibration-method" << error_msg;
    }
    if (calibration_batch_size_.has_value()) {
      LOG(WARNING) << "--calibration-batch-size" << error_msg;
    }
    if (calibration_image_dir_.has_value()) {
      LOG(WARNING) << "--calibration-image-dir" << error_msg;
    }
    if (calibration_percentile_.has_value()) {
      LOG(WARNING) << "--calibration-percentile" << error_msg;
    }
    if (dump_calibrated_model_) {
      LOG(WARNING) << "--dump-calibrated-model" << error_msg;
    }
    if (dump_calibration_table_) {
      LOG(WARNING) << "--dump-calibration-table" << error_msg;
    }
    if (calibrated_model_dump_path_.has_value()) {
      LOG(WARNING) << "--calibrated-model-dump-path" << error_msg;
    }
    if (calibration_table_dump_path_.has_value()) {
      LOG(WARNING) << "--calibration-table-dump-path" << error_msg;
    }
  }

  if (do_collect_) {
    result = CheckStringFilePathWritable("--collect-quant-max-dump-path",
                                         collect_quant_max_path_.value());
    if (!result) return tl::make_unexpected(result.error());
    result = CheckArgExist("--collect-image-dir", collect_image_dir_,
                           "required from --collect");
    if (!result) return tl::make_unexpected(result.error());
    result =
        CheckStringDirectory("--collect-image-dir", collect_image_dir_.value());
    if (!result) return tl::make_unexpected(result.error());
  } else {
    const string error_msg = " is ignored: --collect=false";
    if (collect_quant_max_path_.has_value()) {
      LOG(WARNING) << "--collect-quant-max-dump-path" << error_msg;
    }
    if (collect_image_dir_.has_value()) {
      LOG(WARNING) << "--collect-image-dir" << error_msg;
    }
  }

  // quant-simulator
  if (do_quant_) {
    result = CheckArgExist("--quant-simulator", quant_simulator_,
                           "required from --quant");
    if (!result) return make_unexpected(result.error());
    const vector<string> quant_simulator_options{"x220", "x330"};
    result = CheckArgInList("--quant-simulator", quant_simulator_.value(),
                            quant_simulator_options);
    if (!result) return make_unexpected(result.error());
    if (quant_simulator_.has_value() && quant_simulator_.value() == "x330") {
      if (quant_cfg_path_.has_value()) {
        result = CheckStringFilePathReadable("--quant-cfg-path",
                                             quant_cfg_path_.value());
        if (!result) return tl::make_unexpected(result.error());
      }
      if (quant_updated_ebias_dump_path_.has_value()) {
        if (!quant_max_path_.has_value()) {
          LOG(WARNING) << "--quant-updated-ebias-dump-path is ignored because "
                          "--quant-max-path is not provided";
        }
        result =
            CheckStringFilePathWritable("--quant-updated-ebias-dump-path",
                                        quant_updated_ebias_dump_path_.value());
        if (!result) return tl::make_unexpected(result.error());
      }
      // --quant-max-path can be created by collect. so do not check here.
    }
  } else {
    const std::string error_msg = " is ignored: --quant=false";
    if (quant_simulator_.has_value()) {
      LOG(WARNING) << "--quant-simulator" << error_msg;
    }
    if (quant_cfg_path_.has_value()) {
      LOG(WARNING) << "--quant-cfg-path" << error_msg;
    }
    if (quant_max_path_.has_value()) {
      LOG(WARNING) << "--quant-max-path" << error_msg;
    }
    if (quant_updated_ebias_dump_path_.has_value()) {
      LOG(WARNING) << "--quant-updated-ebias-dump-path" << error_msg;
    }
  }

  // image-path
  if (do_infer_) {
    result =
        CheckArgExist("--image-path", image_path_, "required from --infer");
    if (!result) return make_unexpected(result.error());
    result = CheckStringFilePathReadable("--image-path", image_path_.value());
    if (!result) return make_unexpected(result.error());
  } else {
    const string error_msg = " is ignored: --infer=false";
    if (image_path_.has_value()) {
      LOG(WARNING) << "--image-path" << error_msg;
    }
  }

  // validation-image-dir
  if (do_valid_) {
    result = CheckArgExist("--validation-image-dir", validation_image_dir_,
                           "required from --valid");
    if (!result) return make_unexpected(result.error());
    result = CheckStringDirectory("validation image dir",
                                  validation_image_dir_.value());
    if (!result) return make_unexpected(result.error());
  } else {
    const string error_msg = " is ignored: --valid=false";
    if (validation_image_dir_.has_value()) {
      LOG(WARNING) << "--validation-image-dir" << error_msg;
    }
  }
  return {};
}

expected<void, SimulatorError> SCOPE::CheckCalibArguments() {
  using spgraph_simulator::calibration::CalibrationMethod;
  // calibration-method
  auto result = CheckArgExist("--calibration-method", calibration_method_,
                              "required from --calib");
  if (!result) return make_unexpected(result.error());
  const vector<CalibrationMethod> calibration_methods{
      CalibrationMethod::kMax, CalibrationMethod::kPercentile,
      CalibrationMethod::kEntropy, CalibrationMethod::kEntropy2};
  result = CheckArgInList("--calibration-method", calibration_method_.value(),
                          calibration_methods);
  if (!result) return make_unexpected(result.error());

  // calibration-num-batches
  result = CheckArgExist("--calibration-batch-size", calibration_batch_size_,
                         "use default value 1. only the first batch is used to "
                         "set the histogram range");
  if (!result) {
    calibration_batch_size_ = 1;
  }

  // calibration-image-dir
  result = CheckArgExist("--calibration-image-dir", calibration_image_dir_,
                         "required from --calib");
  if (!result) return make_unexpected(result.error());
  result = CheckStringDirectory("--calibration-image-dir",
                                calibration_image_dir_.value());
  if (!result) return make_unexpected(result.error());

  // calibration-percentile
  if (calibration_method_ ==
      spgraph_simulator::calibration::CalibrationMethod::kPercentile) {
    result = CheckArgExist("--calibration-percentile", calibration_percentile_,
                           "required from --calibration-method=percentile");
    if (!result) return make_unexpected(result.error());
    if (calibration_percentile_.value() < 0.0f ||
        calibration_percentile_.value() > 1.0f) {
      LOG(ERROR) << "Invalid value for --calibration-percentile: should be "
                    "within [0, 1]";
      return make_unexpected(SimulatorError::kArgumentsParsingError);
    }
  }

  // calibrated-model-dump-path
  if (dump_calibrated_model_) {
    result = CheckArgExist("--calibrated-model-dump-path",
                           calibrated_model_dump_path_,
                           "required from --dump-calibrated-model");
    if (!result) return make_unexpected(result.error());
    result = CheckStringFilePathWritable("--calibrated-model-dump-path",
                                         calibrated_model_dump_path_.value());
    if (!result) return make_unexpected(result.error());
  } else {
    result = CheckArgNotExist("--calibrated-model-dump-path",
                              calibrated_model_dump_path_,
                              "required from --dump-calibrated-model=false");
    if (!result) return make_unexpected(result.error());
  }

  // calibration-table-dump-path
  if (dump_calibration_table_) {
    result = CheckArgExist("--calibration-table-dump-path",
                           calibration_table_dump_path_,
                           "required from --dump-calibration-table");
    if (!result) return make_unexpected(result.error());
    result = CheckStringFilePathWritable("--calibration-table-dump-path",
                                         calibration_table_dump_path_.value());
    if (!result) return make_unexpected(result.error());
  } else {
    result = CheckArgNotExist("--calibration-table-dump-path",
                              calibration_table_dump_path_,
                              "required from --dump-calibration-table=false");
    if (!result) return make_unexpected(result.error());
  }
  return {};
}

void SCOPE::do_calib(bool value) { do_calib_ = value; }

bool SCOPE::do_calib() { return do_calib_; }

void SCOPE::do_quant(bool value) { do_quant_ = value; }

bool SCOPE::do_quant() { return do_quant_; }

void SCOPE::do_infer(bool value) { do_infer_ = value; }

bool SCOPE::do_infer() { return do_infer_; }

void SCOPE::do_valid(bool value) { do_valid_ = value; }

bool SCOPE::do_valid() { return do_valid_; }

void SCOPE::backend(string value) { backend_ = value; }

string &SCOPE::backend() { return backend_; }

void SCOPE::graph_type(string value) { graph_type_ = value; }

string &SCOPE::graph_type() { return graph_type_; }

void SCOPE::model_path(string value) { model_path_ = value; }

string &SCOPE::model_path() { return model_path_; }

void SCOPE::preprocess_config_path(optional<string> value) {
  preprocess_config_path_ = value;
}

optional<string> &SCOPE::preprocess_config_path() {
  return preprocess_config_path_;
}

void SCOPE::dump_level(string value) { dump_level_ = value; }

string &SCOPE::dump_level() { return dump_level_; }

void SCOPE::dump_dir(optional<string> value) { dump_dir_ = value; }

optional<string> &SCOPE::dump_dir() { return dump_dir_; }

void SCOPE::calibration_method(string value) {
  using spgraph_simulator::calibration::CalibrationMethod;
  if (value == "max") {
    calibration_method_ = CalibrationMethod::kMax;
  } else if (value == "percentile") {
    calibration_method_ = CalibrationMethod::kPercentile;
  } else if (value == "entropy") {
    calibration_method_ = CalibrationMethod::kEntropy;
  } else if (value == "entropy2") {
    calibration_method_ = CalibrationMethod::kEntropy2;
  } else {
    LOG(ERROR) << "unexpected CalibrationMethod: " << value;
    calibration_method_ = std::nullopt;
  }
}

void SCOPE::calibration_image_dir(optional<string> value) {
  calibration_image_dir_ = value;
}

optional<string> &SCOPE::calibration_image_dir() {
  return calibration_image_dir_;
}

void SCOPE::calibration_percentile(optional<float> value) {
  calibration_percentile_ = value;
}

optional<float> SCOPE::calibration_percentile() {
  return calibration_percentile_;
}

void SCOPE::dump_calibrated_model(bool value) {
  dump_calibrated_model_ = value;
}

bool SCOPE::dump_calibrated_model() { return dump_calibrated_model_; }

void SCOPE::dump_calibration_table(bool value) {
  dump_calibration_table_ = value;
}

bool SCOPE::dump_calibration_table() { return dump_calibration_table_; }

void SCOPE::calibrated_model_dump_path(optional<string> value) {
  calibrated_model_dump_path_ = value;
}

optional<string> &SCOPE::calibrated_model_dump_path() {
  return calibrated_model_dump_path_;
}

void SCOPE::calibration_table_dump_path(optional<string> value) {
  calibration_table_dump_path_ = value;
}

optional<string> &SCOPE::calibration_table_dump_path() {
  return calibration_table_dump_path_;
}

void SCOPE::validation_image_dir(optional<string> value) {
  validation_image_dir_ = value;
}

optional<string> &SCOPE::validation_image_dir() {
  return validation_image_dir_;
}

void SCOPE::image_path(optional<string> value) { image_path_ = value; }

optional<string> &SCOPE::image_path() { return image_path_; }

template <typename T>
expected<void, SimulatorError> SCOPE::CheckArgExist(const string &option_name,
                                                    const optional<T> &argument,
                                                    const string &msg) {
  if (argument.has_value()) {
    return {};
  }
  LOG(ERROR) << "No value for " << option_name << " was given: " << msg;
  return make_unexpected(SimulatorError::kArgumentsParsingError);
}

template <typename T>
expected<void, SimulatorError> SCOPE::CheckArgNotExist(
    const string &option_name, const optional<T> &argument, const string &msg) {
  if (!argument.has_value()) {
    return {};
  }
  LOG(ERROR) << option_name << " should not be given: " << msg;
  return make_unexpected(SimulatorError::kArgumentsParsingError);
}

template <typename T>
expected<void, SimulatorError> SCOPE::CheckArgInList(
    const string &option_name, const T &argument, const vector<T> &valid_args) {
  if (std::find(valid_args.begin(), valid_args.end(), argument) !=
      valid_args.end()) {
    return {};
  }
  LOG(ERROR) << "Invalid value for " << option_name << ": " << argument;
  return make_unexpected(SimulatorError::kArgumentsParsingError);
}

expected<void, SimulatorError> SCOPE::CheckStringFilePathReadable(
    const string &option_name, const string &path) {
  if (!CheckFilePathReadable(path)) {
    const string error_msg =
        "Invalid file path for " + option_name + ": " + path;
    LOG(ERROR) << error_msg;
    return make_unexpected(SimulatorError::kArgumentsParsingError);
  }
  return {};
}

expected<void, SimulatorError> SCOPE::CheckStringFilePathWritable(
    const string &option_name, const string &path) {
  if (!CheckFilePathWritable(path)) {
    const string error_msg =
        "Invalid file path for " + option_name + ": " + path;
    LOG(ERROR) << error_msg;
    return make_unexpected(SimulatorError::kArgumentsParsingError);
  }
  return {};
}

expected<void, SimulatorError> SCOPE::CheckStringDirectory(
    const string &option_name, const string &path) {
  if (!CheckDirectoryExist(path)) {
    const string error_msg =
        "Invalid directory path for " + option_name + ": " + path;
    LOG(ERROR) << error_msg;
    return make_unexpected(SimulatorError::kArgumentsParsingError);
  }
  return {};
}
