#include "backends/delegate/x330_quantization_delegate.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "datatype.hpp"
#include "enums/error.hpp"
#include "factory.hpp"
#include "glog/logging.h"
#include "network/network.hpp"
#include "tl/expected.hpp"
#include "utility.hpp"
#include "x330/aixv_base.h"
#include "x330/aixv_float.h"
#include "x330/aixv_utils.h"
#include "x330/quant_config.hpp"
#include "x330/x330_operation.hpp"

namespace quantization {

X330QuantizationDelegate::X330QuantizationDelegate(Backend& parent,
                                                   Arguments& args)
    : parent_(parent),
      dump_(args),
      cfg_file_path_{args.quant_cfg_path()},
      max_file_path_{args.quant_max_path()},
      updated_ebias_dump_file_path_{args.quant_updated_ebias_dump_path()} {}

tl::expected<void, SimulatorError> X330QuantizationDelegate::Quantize(
    std::unique_ptr<Network>& network) {
  LOG(INFO) << "Quantize Started\n";
  struct timespec start_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);

  LOG(INFO) << "ParseConfigFile Started";
  if (cfg_file_path_.has_value()) {
    auto result = ParseConfigFile(cfg_file_path_.value(), network);
    if (!result) return tl::make_unexpected(result.error());
  } else {
    LOG(INFO) << "The quant config file path does not exist. Use default quant "
                 "configs";
    InitDefaultQuantConfig(network);
  }
  LOG(INFO) << "ParseConfigFile Finished";
  LOG(INFO) << "PrepareQuantOperation Started";
  PrepareQuantOperation(network);
  LOG(INFO) << "PrepareQuantOperation Finished";

  PrintElapsedTime(start_time);
  LOG(INFO) << "Quantize Finished\n";
  return {};
}

namespace {
template <typename T>
tl::expected<T, SimulatorError> FindKeys(
    const std::unordered_map<std::string, T>& map, const std::string& key) {
  const auto& it = map.find(key);
  if (it != map.end()) {
    const auto& [_, value] = *it;
    return value;
  }
  LOG(ERROR) << "Unknown token in quant config file : " << key;
  return tl::make_unexpected(SimulatorError::kQuantizationError);
}
}  // namespace

tl::expected<void, SimulatorError> X330QuantizationDelegate::ParseConfigFile(
    const std::filesystem::path& file_path, std::unique_ptr<Network>& network) {
  const std::unordered_map<std::string, dty::DataType> dtype_names = {
      {"NF8", dty::DataType::NF8},     {"NF8U", dty::DataType::NF8U},
      {"NF8E", dty::DataType::NF8E},   {"NF9E", dty::DataType::NF9E},
      {"NF10", dty::DataType::NF10},   {"NF10U", dty::DataType::NF10U},
      {"NF12", dty::DataType::NF12},   {"NF13", dty::DataType::NF13},
      {"NF13E", dty::DataType::NF13E}, {"NF14", dty::DataType::NF14},
      {"NF14E", dty::DataType::NF14E}, {"NF15E", dty::DataType::NF15E},
      {"NF16", dty::DataType::NF16},   {"BF16", dty::DataType::BF16},
      {"FP32", dty::DataType::FP32}};
  const std::unordered_map<std::string, x330::RoundMode> rmode_names = {
      {"EVEN", x330::RoundMode::ROUND_TO_NEAREST_EVEN},
      {"UP", x330::RoundMode::ROUND_TO_NEAREST_UP},
      {"ZERO", x330::RoundMode::ROUND_TO_ZERO}};
  const std::unordered_map<std::string, x330::QuantConfig::FcalMode>
      fcalib_names = {{"NONE", x330::QuantConfig::FcalMode::FCAL_NONE},
                      {"SET", x330::QuantConfig::FcalMode::FCAL_SET},
                      {"ADD", x330::QuantConfig::FcalMode::FCAL_ADD},
                      {"MIN", x330::QuantConfig::FcalMode::FCAL_MIN}};
  const std::unordered_map<std::string, x330::QuantConfig::WcalMode>
      wcalib_names = {{"NONE", x330::QuantConfig::WcalMode::WCAL_NONE},
                      {"LAYER", x330::QuantConfig::WcalMode::WCAL_LAYER},
                      {"FILTER", x330::QuantConfig::WcalMode::WCAL_FILTER}};

  auto parse_cfg_line = [&](const std::vector<std::string>& tokens)
      -> tl::expected<x330::QuantConfig, SimulatorError> {
    x330::QuantConfig cfg;

    std::vector<dty::DataType> dtypes;
    dtypes.reserve(5);
    for (int i = 0; i < 5; i++) {
      auto result = FindKeys(dtype_names, tokens.at(1 + i));
      if (!result) return tl::make_unexpected(result.error());
      dtypes.push_back(result.value());
    }

    cfg.num_samples = 0;
    cfg.input_max = 0.0;
    cfg.actin_max = 0.0;
    cfg.output_max = 0.0;

    cfg.input_dtype = dtypes.at(0);
    cfg.actin_dtype = dtypes.at(1);
    cfg.output_dtype = dtypes.at(2);
    cfg.weight_dtype = dtypes.at(3);
    cfg.bias_dtype = dtypes.at(4);

    cfg.input_ebias = std::stoi(tokens.at(6));
    cfg.actin_ebias = std::stoi(tokens.at(7));
    cfg.output_ebias = std::stoi(tokens.at(8));
    cfg.weight_ebias = std::stoi(tokens.at(9));
    cfg.bias_ebias = std::stoi(tokens.at(10));

    std::vector<x330::RoundMode> rmodes;
    rmodes.reserve(5);
    for (int i = 0; i < 5; i++) {
      auto result = FindKeys(rmode_names, tokens.at(11 + i));
      if (!result) return tl::make_unexpected(result.error());
      rmodes.push_back(result.value());
    }

    cfg.input_rmode = rmodes.at(0);
    cfg.actin_rmode = rmodes.at(1);
    cfg.output_rmode = rmodes.at(2);
    cfg.weight_rmode = rmodes.at(3);
    cfg.bias_rmode = rmodes.at(4);

    std::vector<x330::QuantConfig::FcalMode> fcalibs;
    fcalibs.reserve(3);
    for (int i = 0; i < 3; i++) {
      auto result = FindKeys(fcalib_names, tokens.at(16 + i));
      if (!result) return tl::make_unexpected(result.error());
      fcalibs.push_back(result.value());
    }

    cfg.input_calib = fcalibs.at(0);
    cfg.actin_calib = fcalibs.at(1);
    cfg.output_calib = fcalibs.at(2);

    auto wcalib_result = FindKeys(wcalib_names, tokens.at(19));
    if (!wcalib_result) return tl::make_unexpected(wcalib_result.error());
    cfg.weight_calib = wcalib_result.value();

    cfg.actfn_lut = static_cast<bool>(std::stoi(tokens.at(20)));
    return cfg;
  };

  std::unordered_map<int, std::vector<std::string>> parsed_cfg;
  {
    auto cfg_file = std::ifstream{file_path};
    std::string line;
    std::string token;
    int id;
    while (std::getline(cfg_file, line)) {
      std::vector<std::string> tokens;
      tokens.reserve(21);
      std::istringstream iss(line);
      while (iss >> token) {
        if (token == "#") break;
        tokens.push_back(token);
      }
      if (tokens.size() == 21) {
        id = std::stoi(tokens.at(0));
        parsed_cfg[id] = tokens;
      }
    }
  }

  x330::QuantConfig default_cfg;
  if (parsed_cfg.find(-2) != parsed_cfg.end()) {
    const auto& parsed_cfg_line = parsed_cfg.at(-2);
    auto result = parse_cfg_line(parsed_cfg_line);
    if (!result) return tl::make_unexpected(result.error());
    default_cfg = result.value();
  } else {
    LOG(ERROR) << "The default value does not exist in the quant cfg file.";
    return tl::make_unexpected(SimulatorError::kQuantizationError);
  }

  x330::QuantConfig input_cfg;
  if (parsed_cfg.find(-1) != parsed_cfg.end()) {
    const auto& parsed_cfg_line = parsed_cfg.at(-1);
    auto result = parse_cfg_line(parsed_cfg_line);
    if (!result) return tl::make_unexpected(result.error());
    input_cfg = result.value();
  } else {
    input_cfg = x330::QuantConfig{default_cfg};
  }

  auto& input_layer = network->input_layer();
  input_layer.x330_quant_config(std::make_shared<x330::QuantConfig>(input_cfg));

  for (auto& layer : network->layers()) {
    const auto id = layer.id();
    if (parsed_cfg.find(id) != parsed_cfg.end()) {
      const auto& parsed_cfg_line = parsed_cfg.at(id);
      auto result = parse_cfg_line(parsed_cfg_line);
      if (!result) return tl::make_unexpected(result.error());
      layer.x330_quant_config(
          std::make_shared<x330::QuantConfig>(result.value()));
    } else {
      x330::QuantConfig default_cfg_cpy{default_cfg};
      layer.x330_quant_config(
          std::make_shared<x330::QuantConfig>(default_cfg_cpy));
    }
  }

  constexpr x330::QuantConfig::FcalMode fcal_none =
      x330::QuantConfig::FcalMode::FCAL_NONE;
  auto update_ebias = [](uint32_t maxv_u, dty::DataType dtype,
                         x330::QuantConfig::FcalMode calib_mode,
                         const int old_exp_bias) {
    union {
      uint32_t u;
      float f;
    } tmp = {maxv_u};
    float maxv_f = tmp.f;
    const float base_max = x330::GetDtypeMax(dtype);
    const int extra_exp_bias =
        -std::ceil(std::log2(static_cast<double>(maxv_f) / base_max));

    if (calib_mode == x330::QuantConfig::FcalMode::FCAL_SET) {
      return extra_exp_bias;
    } else if (calib_mode == x330::QuantConfig::FcalMode::FCAL_ADD) {
      return old_exp_bias + extra_exp_bias;
    } else if (calib_mode == x330::QuantConfig::FcalMode::FCAL_MIN) {
      return std::min(old_exp_bias, extra_exp_bias);
    } else {
      return old_exp_bias;
    }
  };

  bool reads_max_file = false;

  {
    const auto& quant_cfg = network->input_layer().x330_quant_config();
    reads_max_file |= quant_cfg.input_calib != fcal_none;
    reads_max_file |= quant_cfg.actin_calib != fcal_none;
    reads_max_file |= quant_cfg.output_calib != fcal_none;
  }

  for (auto& layer : network->layers()) {
    const auto& quant_cfg = layer.x330_quant_config();
    reads_max_file |= quant_cfg.input_calib != fcal_none;
    reads_max_file |= quant_cfg.actin_calib != fcal_none;
    reads_max_file |= quant_cfg.output_calib != fcal_none;
  }

  if (max_file_path_.has_value() && reads_max_file) {
    std::ifstream quant_max_file{max_file_path_.value()};
    if (!quant_max_file.is_open()) {
      LOG(ERROR) << "Can't open " << max_file_path_.value();
      return tl::make_unexpected(SimulatorError::kFileReadError);
    }
    LOG(INFO) << "Use " << max_file_path_.value()
              << " for fcalib quantization.";
    std::string line;
    std::string token;
    int id;
    bool is_first_line = true;
    while (std::getline(quant_max_file, line)) {
      std::vector<std::string> tokens;
      tokens.reserve(4);
      std::istringstream iss(line);
      while (iss >> token) {
        if (token == "#") break;
        tokens.push_back(token);
      }
      if (tokens.size() == 4) {
        id = std::stoi(tokens.at(0));
        auto& layer = id == -1 ? network->input_layer() : network->layers(id);
        auto& quant_cfg = layer.x330_quant_config();
        bool do_fcalib = false;
        {
          do_fcalib |= quant_cfg.input_calib != fcal_none;
          do_fcalib |= quant_cfg.actin_calib != fcal_none;
          do_fcalib |= quant_cfg.output_calib != fcal_none;
        }

        const auto prev_ebiases =
            std::make_tuple(quant_cfg.input_ebias, quant_cfg.actin_ebias,
                            quant_cfg.output_ebias);

        if (do_fcalib) {
          quant_cfg.input_ebias = update_ebias(
              std::stoi(tokens.at(1), nullptr, 16), quant_cfg.input_dtype,
              quant_cfg.input_calib, quant_cfg.input_ebias);
          quant_cfg.actin_ebias = update_ebias(
              std::stoi(tokens.at(2), nullptr, 16), quant_cfg.actin_dtype,
              quant_cfg.actin_calib, quant_cfg.actin_ebias);
          quant_cfg.output_ebias = update_ebias(
              std::stoi(tokens.at(3), nullptr, 16), quant_cfg.output_dtype,
              quant_cfg.output_calib, quant_cfg.output_ebias);
        }

        dump_.DumpX330UpdatedEbiases(
            layer, prev_ebiases, updated_ebias_dump_file_path_, is_first_line);
        is_first_line = false;
      }
    }
  }
  return {};
}

tl::expected<void, SimulatorError>
X330QuantizationDelegate::PrepareQuantOperation(
    std::unique_ptr<Network>& network) {
  const int num_layers = network->num_layers();
  for (int i = 0; i < num_layers; i++) {
    Layer& layer = network->layers(i);
    const int num_sublayers = network->num_operations(i);
    for (int j = 0; j < num_sublayers; j++) {
      std::string operation_name = layer.operation_types(j);

      auto quant_op =
          Factory<x330::X330Operation>::CreateInstance(operation_name);
      if (quant_op == nullptr) {
        continue;
      }
      if (operation_name == "Convolution") {
        dump_.DumpX330LayerFilterFP(layer);
      }

      quant_op->PrepareQuantOperation(network, i);

      if (operation_name == "Convolution" || operation_name == "Connected") {
        dump_.DumpX330LayerFilter(layer);
      }
    }
  }
  return {};
}

void X330QuantizationDelegate::InitDefaultQuantConfig(
    std::unique_ptr<Network>& network) {
  x330::QuantConfig cfg;
  {
    cfg.num_samples = 0;
    cfg.input_max = 0.0;
    cfg.actin_max = 0.0;
    cfg.output_max = 0.0;

    cfg.input_dtype = dty::DataType::FP32;
    cfg.actin_dtype = dty::DataType::NF16;
    cfg.output_dtype = dty::DataType::NF8;
    cfg.weight_dtype = dty::DataType::NF8;
    cfg.bias_dtype = dty::DataType::NF16;

    cfg.input_ebias = 4;
    cfg.actin_ebias = 0;
    cfg.output_ebias = 4;
    cfg.weight_ebias = 4;
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
  auto& input_layer = network->input_layer();
  input_layer.x330_quant_config(std::make_shared<x330::QuantConfig>(cfg));

  for (auto& layer : network->layers()) {
    x330::QuantConfig cfg_cpy{cfg};
    layer.x330_quant_config(std::make_shared<x330::QuantConfig>(cfg_cpy));
  }
}
}  // namespace quantization
