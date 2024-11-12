#include "backends/delegate/x330_collect_delegate.hpp"

#include <fstream>
#include <iomanip>
#include <memory>
#include <string>

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "datatype.hpp"
#include "enums/dump.hpp"
#include "enums/error.hpp"
#include "factory.hpp"
#include "glog/logging.h"
#include "network/network.hpp"
#include "tl/expected.hpp"
#include "utility.hpp"
#include "x330/aixv_utils.h"
#include "x330/x330_operation.hpp"

namespace collect {
X330CollectDelegate::X330CollectDelegate(Backend &parent, Arguments &args)
    : parent_(parent) {
  if (args.collect_quant_max_path().has_value()) {
    GetAbsoluteFilePath(quant_max_path_, args.collect_quant_max_path().value());
  } else {
    std::string dump_path =
        args.dump_dir().value_or("dump").append("quant.max");
    GetAbsoluteFilePath(quant_max_path_, dump_path);
  }
  GetImageFilePaths(collect_image_paths_, args.collect_image_dir().value());
  dump_level_ = GetDumpLevel(args.dump_level());
}

tl::expected<void, SimulatorError> X330CollectDelegate::Collect(
    std::unique_ptr<Network> &network) {
  LOG(INFO) << "InitFP32QuantConfig";

  InitFPQuantConfig(network);

  LOG(INFO) << "Collect Started";
  struct timespec start_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);

  CollectMaxes(network);

  DumpQuantMax(network, quant_max_path_, dump_level_);

  PrintElapsedTime(start_time);
  LOG(INFO) << "Collect Finished";
  return {};
}

tl::expected<void, SimulatorError> X330CollectDelegate::CollectMaxes(
    std::unique_ptr<Network> &network) {
  using x330::X330Operation;
  int idx_image = 0;
  for (const auto &image_path : collect_image_paths_) {
    idx_image++;
    PrintBatchFlags(idx_image, collect_image_paths_.size());

    auto input_tensor =
        parent_.GetInputImageTensor(image_path, dty::DataType::FP32).value();
    InferenceContext input_ctx{*network, input_tensor};
    input_ctx.SetLayerContext({-1}, 0, 1);
    auto input_ops = Factory<x330::X330Operation>::CreateInstance("Input");
    input_ops->Forward(network->input_layer(), input_ctx);
    const auto converted = std::move(*input_ctx.GetLayerOutputTensor(0));

    InferenceContext ctx{*network, converted};

    for (int idx_layer = 0; idx_layer < network->num_layers(); idx_layer++) {
      auto &layer = network->layers(idx_layer);
      ctx.SetLayerContext(layer.predecessors(), idx_layer,
                          layer.operation_types().size());
      for (const auto &op_name : layer.operation_types()) {
        auto operation = Factory<x330::X330Operation>::CreateInstance(op_name);
        if (operation == nullptr) {
          DLOG(ERROR) << "Failed to create operation: " << op_name;
        }
        operation->Forward(layer, ctx);
      }
      ctx.EraseUsedTensors();
    }
  }
  return {};
}

tl::expected<void, SimulatorError> X330CollectDelegate::DumpQuantMax(
    std::unique_ptr<Network> &network, const std::string &file_path,
    const DumpLevel level) {
  // aix_finalize_net
  TruncateX330LayerMax(file_path);
  auto &layers = network->layers();
  for (auto it = layers.rbegin(); it != layers.rend(); it++) {
    DumpX330LayerMax(*it, file_path, level);
  }
  DumpX330LayerMax(network->input_layer(), file_path, level);
  return {};
}

void X330CollectDelegate::InitFPQuantConfig(std::unique_ptr<Network> &network) {
  x330::QuantConfig cfg;
  {
    cfg.num_samples = 0;
    cfg.input_max = 0.0;
    cfg.actin_max = 0.0;
    cfg.output_max = 0.0;

    cfg.input_dtype = dty::DataType::FP32;
    cfg.actin_dtype = dty::DataType::FP32;
    cfg.output_dtype = dty::DataType::FP32;
    cfg.weight_dtype = dty::DataType::FP32;
    cfg.bias_dtype = dty::DataType::FP32;

    cfg.input_ebias = 0;
    cfg.actin_ebias = 0;
    cfg.output_ebias = 0;
    cfg.weight_ebias = 0;
    cfg.bias_ebias = 0;

    cfg.input_rmode = x330::RoundMode::ROUND_TO_NEAREST_EVEN;
    cfg.actin_rmode = x330::RoundMode::ROUND_TO_NEAREST_EVEN;
    cfg.output_rmode = x330::RoundMode::ROUND_TO_NEAREST_EVEN;
    cfg.weight_rmode = x330::RoundMode::ROUND_TO_NEAREST_EVEN;
    cfg.bias_rmode = x330::RoundMode::ROUND_TO_NEAREST_EVEN;

    cfg.input_calib = x330::QuantConfig::FcalMode::FCAL_NONE;
    cfg.actin_calib = x330::QuantConfig::FcalMode::FCAL_NONE;
    cfg.output_calib = x330::QuantConfig::FcalMode::FCAL_NONE;

    cfg.weight_calib = x330::QuantConfig::WcalMode::WCAL_NONE;

    cfg.actfn_lut = false;
  }
  auto &input_layer = network->input_layer();
  input_layer.x330_quant_config(std::make_shared<x330::QuantConfig>(cfg));

  for (auto &layer : network->layers()) {
    x330::QuantConfig cfg_cpy{cfg};
    layer.x330_quant_config(std::make_shared<x330::QuantConfig>(cfg_cpy));
  }
}

void X330CollectDelegate::TruncateX330LayerMax(const std::string &file_path) {
  std::ofstream dump_file_path{file_path, std::ios::trunc};
}

void X330CollectDelegate::DumpX330LayerMax(Layer &layer,
                                           const std::string &file_path,
                                           DumpLevel level) {
  if (level == DumpLevel::DUMP_NONE) return;

  std::ofstream dump_file{file_path, std::ios::app};
  if (!dump_file.is_open()) {
    // TODO: handle error
  }

  const auto &cfg = layer.x330_quant_config();
  const float imax = cfg.input_max / cfg.num_samples;
  const float amax = cfg.actin_max / cfg.num_samples;
  const float omax = cfg.output_max / cfg.num_samples;

  union {
    float f;
    uint32_t u;
  } tmp;
  auto get_fp32_bits = [&tmp](float fp32) {
    tmp.f = fp32;
    return tmp.u;
  };

  std::stringstream output;
  output << std::setw(3) << layer.id();

  output << " " << std::uppercase << std::hex << std::setfill('0')
         << std::setw(8) << get_fp32_bits(imax);

  output << " " << std::uppercase << std::hex << std::setfill('0')
         << std::setw(8) << get_fp32_bits(amax);

  output << " " << std::uppercase << std::hex << std::setfill('0')
         << std::setw(8) << get_fp32_bits(omax);

  output << " # " << std::to_string(imax) << " " << std::to_string(amax) << " "
         << std::to_string(omax) << "\n";

  dump_file << output.str();
}
}  // namespace collect
