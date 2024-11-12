#include "backends/delegate/default_calibration_delegate.hpp"

#define SCOPE DefaultCalibrationDelegate

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "backends/delegate/calibration_dump_helper.hpp"
#include "calibration/calibrator_factory.hpp"
#include "calibration/histogram_calibrator.hpp"
#include "calibration/max.hpp"
#include "datatype.hpp"
#include "enums/error.hpp"
#include "factory.hpp"
#include "glog/logging.h"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "tl/expected.hpp"
#include "utility.hpp"

SCOPE::DefaultCalibrationDelegate(Backend &parent, Arguments &args)
    : parent_(parent), dump_(args) {
  // Parse Arguments
  input_type_ = args.input_type;
  graph_type_ = args.graph_type();
  graph_binary_path_ = args.model_path();
  calibration_mode_ = args.calibration_method().value();
  percentile_ = args.calibration_percentile();
  backend_type_ = args.backend();

  switch (input_type_) {
    case Arguments::InputType::kNumpy:
      GetNumpyFilePaths(calibration_image_paths_,
                        args.calibration_image_dir().value());
      break;
    case Arguments::InputType::kImage:
      GetImageFilePaths(calibration_image_paths_,
                        args.calibration_image_dir().value());
      break;
  }

  if (args.calibration_batch_size().has_value()) {
    batch_size_ = args.calibration_batch_size().value();
  } else {
    batch_size_ = 1;
  }
}

tl::expected<void, SimulatorError> SCOPE::Calibrate(
    std::unique_ptr<Network> &network) {
  namespace calib = spgraph_simulator::calibration;

  LOG(INFO) << "Calibrate Started\n";
  struct timespec start_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);

  LOG(INFO) << "Create Calibrator";
  auto factory = calib::CalibratorFactory{};
  auto calibrator =
      factory.GetCalibrator(*network, calibration_mode_, percentile_);

  LOG(INFO) << "Collect Layer Outputs";
  const size_t num_batches = calibration_image_paths_.size() / batch_size_;
  const bool is_exactly_divided =
      calibration_image_paths_.size() % batch_size_ == 0;
  if (!is_exactly_divided) {
    LOG(WARNING) << "The total number of images is not exactly divided by the "
                    "batch size : "
                 << calibration_image_paths_.size() << " % " << batch_size_
                 << " != 0";
  }

  for (int idx_batch = 0;
       idx_batch < (is_exactly_divided ? num_batches : num_batches + 1);
       ++idx_batch) {
    std::vector<Tensor> input_tensors;
    input_tensors.reserve(batch_size_);

    const size_t idx_start = batch_size_ * idx_batch;
    const size_t idx_end = std::min(batch_size_ * (idx_batch + 1),
                                    calibration_image_paths_.size());
    for (int idx_image = idx_start; idx_image < idx_end; idx_image++) {
      DLOG(INFO) << "Image " << idx_image + 1 << " / "
                 << calibration_image_paths_.size() << " ...";
      auto res =
          GetInputTensor(input_type_, calibration_image_paths_.at(idx_image));
      if (res) {
        input_tensors.push_back(std::move(res.value()));
      } else {
        return tl::make_unexpected(res.error());
      }
    }

    auto res = parent_.FuseInputTensors(input_tensors);
    if (res) {
      LOG(INFO) << "Collect " << idx_batch + 1
                << "'th batch : " << idx_end - idx_start << " number of data";
      calibrator->Collect(res.value());
    } else {
      return tl::make_unexpected(res.error());
    }
  }

  LOG(INFO) << "Compute Layer Thresholds";
  // Network's thresholds are also updated
  auto calib_result = calibrator->ComputeRange();

  LOG(INFO) << "Dump Calibration Results";
  dump_.DumpCalibrationTableSapeonFormat(calib_result);

  dump_.DumpCalibratedModel(graph_type_, graph_binary_path_, network);

  PrintElapsedTime(start_time);
  LOG(INFO) << "Calibrate Finished\n";

  // TODO: add error handling logic
  return {};
}

tl::expected<Tensor, SimulatorError> SCOPE::GetInputTensor(
    Arguments::InputType input_type, const std::string &input_file_path) {
  switch (input_type) {
    case Arguments::InputType::kNumpy:
      return parent_.GetInputNumpyTensor(input_file_path);
    case Arguments::InputType::kImage:
      return parent_.GetInputImageTensor(input_file_path, dty::DataType::FP32);
    default:
      LOG(ERROR) << "Unknown Input Type: " << '\n';
      return tl::make_unexpected(SimulatorError::kInvalidDataType);
  }
}
