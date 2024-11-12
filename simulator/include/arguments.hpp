#ifndef ARGUMENTS_HPP
#define ARGUMENTS_HPP

#include <optional>
#include <string>
#include <vector>

#include "enums/calibration.hpp"
#include "enums/error.hpp"
#include "tl/expected.hpp"

class Arguments {
 public:
  tl::expected<void, SimulatorError> CheckArguments();
  enum class InputType { kImage = 0, kNumpy };
  InputType input_type;

  void do_calib(bool value);
  bool do_calib();
  void do_collect(bool value) { do_collect_ = value; }
  bool do_collect() { return do_collect_; }
  void do_quant(bool value);
  bool do_quant();
  void do_infer(bool value);
  bool do_infer();
  void do_valid(bool value);
  bool do_valid();
  void backend(std::string value);
  std::string &backend();
  void graph_type(std::string value);
  std::string &graph_type();
  void model_path(std::string value);
  std::string &model_path();
  void preprocess_config_path(std::optional<std::string> value);
  std::optional<std::string> &preprocess_config_path();
  void dump_level(std::string value);
  std::string &dump_level();
  void dump_dir(std::optional<std::string> value);
  std::optional<std::string> &dump_dir();
  void calibration_method(std::string value);
  void calibration_method(std::nullopt_t nullopt) {
    calibration_method_ = std::nullopt;
  }
  void calibration_method(
      spgraph_simulator::calibration::CalibrationMethod value) {
    calibration_method_ = value;
  }
  std::optional<spgraph_simulator::calibration::CalibrationMethod>
  calibration_method() {
    return calibration_method_;
  }
  void calibration_image_dir(std::optional<std::string> value);
  std::optional<std::string> &calibration_image_dir();
  void calibration_batch_size(std::optional<size_t> value) {
    calibration_batch_size_ = value;
  }
  std::optional<size_t> calibration_batch_size() {
    return calibration_batch_size_;
  }
  void calibration_percentile(std::optional<float> value);
  std::optional<float> calibration_percentile();
  void dump_calibrated_model(bool value);
  bool dump_calibrated_model();
  void dump_calibration_table(bool value);
  bool dump_calibration_table();
  void calibrated_model_dump_path(std::optional<std::string> value);
  std::optional<std::string> &calibrated_model_dump_path();
  void calibration_table_dump_path(std::optional<std::string> value);
  std::optional<std::string> &calibration_table_dump_path();
  void collect_image_dir(std::optional<std::string> value) {
    collect_image_dir_ = value;
  }
  std::optional<std::string> &collect_image_dir() { return collect_image_dir_; }
  void collect_quant_max_path(std::optional<std::string> value) {
    collect_quant_max_path_ = value;
  }
  std::optional<std::string> &collect_quant_max_path() {
    return collect_quant_max_path_;
  }
  void quant_simulator(std::optional<std::string> value) {
    quant_simulator_ = value;
  }
  std::optional<std::string> &quant_simulator() { return quant_simulator_; }
  void quant_cfg_path(std::optional<std::string> value) {
    quant_cfg_path_ = value;
  }
  const std::optional<std::string> &quant_cfg_path() const {
    return quant_cfg_path_;
  }
  void quant_max_path(std::optional<std::string> value) {
    quant_max_path_ = value;
  }
  const std::optional<std::string> &quant_max_path() const {
    return quant_max_path_;
  }
  void quant_updated_ebias_dump_path(std::optional<std::string> value) {
    quant_updated_ebias_dump_path_ = value;
  }
  const std::optional<std::string> &quant_updated_ebias_dump_path() const {
    return quant_updated_ebias_dump_path_;
  }
  void image_path(std::optional<std::string> value);
  std::optional<std::string> &image_path();
  void validation_image_dir(std::optional<std::string> value);
  std::optional<std::string> &validation_image_dir();

 private:
  tl::expected<void, SimulatorError> CheckCalibArguments();
  template <typename T>
  tl::expected<void, SimulatorError> CheckArgExist(
      const std::string &option_name, const std::optional<T> &argument,
      const std::string &msg);
  template <typename T>
  tl::expected<void, SimulatorError> CheckArgNotExist(
      const std::string &option_name, const std::optional<T> &argument,
      const std::string &msg);
  template <typename T>
  tl::expected<void, SimulatorError> CheckArgInList(
      const std::string &option_name, const T &argument,
      const std::vector<T> &valid_args);
  tl::expected<void, SimulatorError> CheckStringFilePathReadable(
      const std::string &option_name, const std::string &path);
  tl::expected<void, SimulatorError> CheckStringFilePathWritable(
      const std::string &option_name, const std::string &path);
  tl::expected<void, SimulatorError> CheckStringDirectory(
      const std::string &option_name, const std::string &path);

  bool do_calib_;
  bool do_collect_;
  bool do_quant_;
  bool do_infer_;
  bool do_valid_;
  std::string backend_;
  std::string graph_type_;
  std::string model_path_;
  std::optional<std::string> preprocess_config_path_;
  std::string dump_level_;
  std::optional<std::string> dump_dir_;
  std::optional<spgraph_simulator::calibration::CalibrationMethod>
      calibration_method_;
  std::optional<std::string> calibration_image_dir_;
  std::optional<size_t> calibration_batch_size_;
  std::optional<float> calibration_percentile_;
  bool dump_calibrated_model_;
  bool dump_calibration_table_;
  std::optional<std::string> calibrated_model_dump_path_;
  std::optional<std::string> calibration_table_dump_path_;
  std::optional<std::string> collect_image_dir_;
  std::optional<std::string> collect_quant_max_path_;
  std::optional<std::string> quant_simulator_;
  std::optional<std::string> quant_cfg_path_;
  std::optional<std::string> quant_max_path_;
  std::optional<std::string> quant_updated_ebias_dump_path_;
  std::optional<std::string> validation_image_dir_;
  std::optional<std::string> image_path_;
};

#endif  // ARGUMENTS_HPP
