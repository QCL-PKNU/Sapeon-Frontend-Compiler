#include "backends/delegate/inference_dump_helper.hpp"

#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string>

#include "arguments.hpp"
#include "datatype.hpp"
#include "dump/dump_factory.hpp"
#include "enums/dump.hpp"
#include "glog/logging.h"
#include "network/tensor.hpp"

namespace fs = std::filesystem;

InferenceDumpHelper::InferenceDumpHelper(Arguments &args)
    : factory_(), path_helper_(args) {
  dump_level_ = GetDumpLevel(args.dump_level());
}

void InferenceDumpHelper::DumpLayerOutput(std::shared_ptr<Tensor> activation,
                                          int idx_layer, DumpLevel dump_level) {
  switch (activation->dtype()) {
    case dty::DataType::FP32:
      DumpLayerOutput<float>(activation, idx_layer, dump_level);
      break;
    case dty::DataType::FP64:
      DumpLayerOutput<double>(activation, idx_layer, dump_level);
      break;
    case dty::DataType::INT8:
      DumpLayerOutput<int8_t>(activation, idx_layer, dump_level);
      break;
    case dty::DataType::UINT8:
      DumpLayerOutput<uint8_t>(activation, idx_layer, dump_level);
      break;
    default:
      LOG(ERROR) << "Unsupported data type for dump layer output!\n";
      exit(1);
  }
}

void InferenceDumpHelper::DumpNetworkInput(std::shared_ptr<Tensor> activation,
                                           DumpLevel dump_level) {
  switch (activation->dtype()) {
    case dty::DataType::FP32:
      DumpNetworkInput<float>(activation, dump_level);
      break;
    case dty::DataType::FP64:
      DumpNetworkInput<double>(activation, dump_level);
      break;
    case dty::DataType::INT8:
      DumpNetworkInput<int8_t>(activation, dump_level);
      break;
    case dty::DataType::UINT8:
      DumpNetworkInput<uint8_t>(activation, dump_level);
      break;
    default:
      LOG(ERROR) << "Unsupported data type for dump network input!\n";
      exit(1);
  }
}

void InferenceDumpHelper::DumpNetworkOutput(std::shared_ptr<Tensor> activation,
                                            DumpLevel dump_level) {
  switch (activation->dtype()) {
    case dty::DataType::FP32:
      DumpNetworkOutput<float>(activation, dump_level);
      break;
    case dty::DataType::FP64:
      DumpNetworkOutput<double>(activation, dump_level);
      break;
    case dty::DataType::INT8:
      DumpNetworkOutput<int8_t>(activation, dump_level);
      break;
    case dty::DataType::UINT8:
      DumpNetworkOutput<uint8_t>(activation, dump_level);
      break;
    default:
      LOG(ERROR) << "Unsupported data type for dump network output!\n";
      exit(1);
  }
}

template <typename Type>
void InferenceDumpHelper::DumpLayerOutput(std::shared_ptr<Tensor> activation,
                                          int idx_layer, DumpLevel dump_level) {
  const std::string path = path_helper_.GetLayerOutputPath(idx_layer);
  auto dump = factory_.GetDump<Type>(dump_level_, DumpFormat::DUMP_SPACE, path);
  dump->DumpTensorNHWC(activation, dump_level, 20);
}

template <typename Type>
void InferenceDumpHelper::DumpNetworkInput(std::shared_ptr<Tensor> tensor,
                                           DumpLevel dump_level) {
  const std::string path = path_helper_.GetFilePath("network.input");
  const std::string path2 = path_helper_.GetFilePath("network.input.binary");
  auto dump = factory_.GetDump<Type>(dump_level_, DumpFormat::DUMP_SPACE, path);
  auto dump2 =
      factory_.GetDump<Type>(dump_level_, DumpFormat::DUMP_BINARY, path2);
  dump->DumpTensor(tensor, dump_level, 20);
  dump2->DumpTensor(tensor, dump_level);
}

template <typename Type>
void InferenceDumpHelper::DumpNetworkOutput(std::shared_ptr<Tensor> output,
                                            DumpLevel dump_level) {
  const std::string path = path_helper_.GetFilePath("network.output");
  auto dump =
      factory_.GetDump<Type>(dump_level_, DumpFormat::DUMP_BINARY, path);
  dump->DumpTensor(output, dump_level);
}

void InferenceDumpHelper::DumpX220Activation(std::shared_ptr<Tensor> activation,
                                             int idx_layer,
                                             DumpLevel dump_level,
                                             float output_threshold) {
  switch (activation->dtype()) {
    case dty::DataType::INT8:
      DumpX220Activation<int8_t>(activation, idx_layer, dump_level,
                                 output_threshold);
      break;
    case dty::DataType::UINT8:
      DumpX220Activation<uint8_t>(activation, idx_layer, dump_level,
                                  output_threshold);
      break;
    default:
      LOG(ERROR) << "Unsupported data type for dump mxc activation!\n";
      exit(1);
  }
}

template <typename Type>
void InferenceDumpHelper::DumpX220Activation(std::shared_ptr<Tensor> activation,
                                             int idx_layer,
                                             DumpLevel dump_level,
                                             float output_threshold) {
  const std::string path_int =
      path_helper_.GetActivationIntOutputPath(idx_layer);
  const std::string path_fp =
      path_helper_.GetActivationFloatOutputPath(idx_layer);
  auto dump_int =
      factory_.GetDump<Type>(dump_level_, DumpFormat::DUMP_OUTPUT, path_int);
  auto dump_fp =
      factory_.GetDump<double>(dump_level_, DumpFormat::DUMP_OUTPUT, path_fp);

  const float kTypeMax = static_cast<float>(std::numeric_limits<Type>::max());
  float scale = kTypeMax / output_threshold;
  std::shared_ptr<Tensor> dequant_tensor = std::make_shared<Tensor>(
      activation->dimension().dims(), dty::DataType::FP64);
  double *dequant_data = dequant_tensor->data<double>();
  Type *data = activation->data<Type>();
  for (int i = 0; i < dequant_tensor->dimension().size(); i++) {
    dequant_data[i] = static_cast<double>(data[i]) / scale;
  }

  dump_int->DumpTensor(activation, dump_level);
  dump_fp->DumpTensor(dequant_tensor, dump_level, 40);
}

void InferenceDumpHelper::DumpX220Input(std::shared_ptr<Tensor> tensor,
                                        DumpLevel dump_level,
                                        float input_threshold) {
  const std::string path1 = path_helper_.GetFilePath("network.input");
  const std::string path2 = path_helper_.GetFilePath("int_input.txt");
  const std::string path3 = path_helper_.GetFilePath("fp_input.txt");

  auto dump1 =
      factory_.GetDump<int8_t>(dump_level_, DumpFormat::DUMP_SPACE, path1);
  auto dump2 =
      factory_.GetDump<int8_t>(dump_level_, DumpFormat::DUMP_OUTPUT, path2);
  auto dump3 =
      factory_.GetDump<double>(dump_level_, DumpFormat::DUMP_OUTPUT, path3);

  const float kINT8Max = static_cast<float>(127);
  double scale = static_cast<double>(kINT8Max / input_threshold);
  std::shared_ptr<Tensor> dequant_tensor =
      std::make_shared<Tensor>(tensor->dimension().dims(), dty::DataType::FP64);
  double *dequant_data = dequant_tensor->data<double>();
  int8_t *data = tensor->data<int8_t>();
  for (int i = 0; i < dequant_tensor->dimension().size(); i++) {
    dequant_data[i] = static_cast<double>(data[i]) / scale;
  }

  dump1->DumpTensorNHWC(tensor, dump_level);
  dump2->DumpTensor(tensor, dump_level);
  dump3->DumpTensor(dequant_tensor, dump_level, 20);
}

// TODO: Delete below methods, move to inference dump helper. Need to refactor
template <typename Type>
void InferenceDumpHelper::DumpTensorDefault(const std::string &file_name,
                                            std::shared_ptr<Tensor> tensor) {
  LOG(ERROR) << "Dump error: unknown tensor data type: "
             << dty::NameOf(tensor->dtype());
}

template <>
void InferenceDumpHelper::DumpTensorDefault<int8_t>(
    const std::string &file_name, std::shared_ptr<Tensor> tensor) {
  const int out_n = tensor->dimension().n();
  const int out_c = tensor->dimension().c();
  const int out_h = tensor->dimension().h();
  const int out_w = tensor->dimension().w();

  FILE *fp = fopen(file_name.c_str(), "w");

  int8_t *data = tensor->data<int8_t>();
  for (int n = 0; n < out_n; ++n) {
    for (int h = 0; h < out_h; ++h) {
      for (int w = 0; w < out_w; ++w) {
        for (int c = 0; c < out_c; ++c) {
          int idx = (out_c * out_h * out_w * n) + (out_h * out_w * c) +
                    (out_w * h) + w;
          fprintf(fp, "%hhd ", data[idx]);
        }
      }
    }
    fprintf(fp, "\n");
  }
}

template <typename Type>
void InferenceDumpHelper::DumpTensorDarknet(const std::string &file_name,
                                            std::shared_ptr<Tensor> tensor) {
  LOG(ERROR) << "Dump error: unknown tensor data type: "
             << dty::NameOf(tensor->dtype());
}

template <>
void InferenceDumpHelper::DumpTensorDarknet<int8_t>(
    const std::string &file_name, std::shared_ptr<Tensor> tensor) {
  const int out_n = tensor->dimension().n();
  const int out_c = tensor->dimension().c();
  const int out_h = tensor->dimension().h();
  const int out_w = tensor->dimension().w();

  FILE *fp = fopen(file_name.c_str(), "w");

  int8_t *data = tensor->data<int8_t>();
  for (int n = 0; n < out_n; ++n) {
    for (int c = 0; c < out_c; ++c) {
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          int idx = (out_c * out_h * out_w * n) + (out_h * out_w * c) +
                    (out_w * h) + w;
          fprintf(fp, "output[%d] = %hhd\n", idx, data[idx]);
        }
      }
    }
  }
}

template <typename Type>
void InferenceDumpHelper::DumpTensorDarknetDequant(
    const std::string &file_name, std::shared_ptr<Tensor> tensor,
    double scale) {
  LOG(ERROR) << "Dump error: unknown tensor data type: "
             << dty::NameOf(tensor->dtype());
}

template <>
void InferenceDumpHelper::DumpTensorDarknetDequant<double>(
    const std::string &file_name, std::shared_ptr<Tensor> tensor,
    double scale) {
  const int out_n = tensor->dimension().n();
  const int out_c = tensor->dimension().c();
  const int out_h = tensor->dimension().h();
  const int out_w = tensor->dimension().w();

  FILE *fp = fopen(file_name.c_str(), "w");

  int8_t *data = tensor->data<int8_t>();
  for (int n = 0; n < out_n; ++n) {
    for (int c = 0; c < out_c; ++c) {
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          int idx = (out_c * out_h * out_w * n) + (out_h * out_w * c) +
                    (out_w * h) + w;
          double val = (double)data[idx] / scale;
          fprintf(fp, "output[%d] = %.20lf\n", idx, val);
        }
      }
    }
  }
}

void InferenceDumpHelper::DumpQuantizedInput(std::shared_ptr<Tensor> tensor,
                                             double scale) {
  DumpTensorDefault<int8_t>("dump/network.input", tensor);
  DumpTensorDarknet<int8_t>("dump/int_input.txt", tensor);
  DumpTensorDarknetDequant<double>("dump/fp_input.txt", tensor, scale);
}

namespace {
auto GetFp32Bits(float fp32) {
  union {
    float f;
    uint32_t u;
  } tmp = {fp32};
  return tmp.u;
};
}  // namespace

void InferenceDumpHelper::DumpX330NetworkInput(std::shared_ptr<Tensor> tensor) {
  if (dump_level_ == DumpLevel::DUMP_NONE) return;
  const bool dumps_debug_files = dump_level_ == DumpLevel::DUMP_DEBUG;

  fs::path dump_path1;
  std::optional<fs::path> dump_path2{std::nullopt};
  std::optional<fs::path> dump_path3{std::nullopt};

  dump_path1 = {path_helper_.GetFilePath("network.input")};
  if (dumps_debug_files) {
    dump_path2 = {path_helper_.GetFilePath("int_input.txt")};
    dump_path3 = {path_helper_.GetFilePath("fp_input.txt")};
  }

  std::ofstream dump_file1{dump_path1, std::ios::out | std::ios::app};
  if (!dump_file1.is_open()) {
    LOG(ERROR) << "Cannot open dump file : " << dump_path1;
    // TODO: handle error
  }

  std::ofstream dump_file2;
  std::ofstream dump_file3;
  if (dumps_debug_files) {
    if (dump_path2.has_value()) {
      dump_file2 = {dump_path2.value(), std::ios::out};
      if (!dump_file2.is_open()) {
        LOG(ERROR) << "Cannot open dump file : " << dump_path2.value();
        // TODO: handle error
      }
    } else {
      LOG(ERROR) << "Cannot create dump file path : int_input.txt";
      // TODO: handle error
    }

    if (dump_path3.has_value()) {
      dump_file3 = {dump_path3.value(), std::ios::out};
      if (!dump_file3.is_open()) {
        LOG(ERROR) << "Cannot open dump file : " << dump_path3.value();
        // TODO: handle error
      }
    } else {
      LOG(ERROR) << "Cannot create dump file path : fp_input.txt";
      // TODO: handle error
    }
  }

  const auto nn = tensor->n();
  const auto cc = tensor->c();
  const auto hh = tensor->h();
  const auto ww = tensor->w();
  const float *tensor_data = tensor->data<float>();

  auto get_element = [&](const float *data, int y, int x, int c) {
    const int idx = (hh * ww * c) + (ww * y) + x;
    return GetFp32Bits(data[idx]);
  };

  for (int b = 0; b < nn; ++b) {
    auto *data = tensor_data + b * (cc * hh * ww);

    for (int y = 0; y < hh; ++y) {
      for (int x = 0; x < ww; ++x) {
        for (int c = 0; c < cc; ++c) {
          uint32_t v = get_element(data, y, x, c);
          dump_file1 << std::hex << std::uppercase << std::setw(8)
                     << std::setfill('0') << v << " ";
        }
      }
    }  // y,x,c
    dump_file1 << "\n";

    // assume n=1
    if (dumps_debug_files) {
      for (int c = 0; c < cc; ++c) {
        for (int y = 0; y < hh; ++y) {
          for (int x = 0; x < ww; ++x) {
            const size_t index = (hh * ww * c) + (ww * y) + x;
            const uint32_t int_val = get_element(data, y, x, c);
            dump_file2 << "output[" << index << "] = " << std::setfill('0')
                       << static_cast<int>(int_val) << "\n";
            dump_file3 << "output[" << index << "] = " << std::fixed
                       << std::setprecision(20) << data[index] << "\n";
          }
        }
      }  // y,x,z
    }
  }
}

void InferenceDumpHelper::DumpX330LayerOutput(const Tensor &output_tensor,
                                              const int idx_layer) {
  // if (!aix_dump_data_enabled()) return;
  if (dump_level_ == DumpLevel::DUMP_NONE) return;
  const bool dumps_debug_files = dump_level_ == DumpLevel::DUMP_DEBUG;

  fs::path dump_path1;
  std::optional<fs::path> dump_path2{std::nullopt};
  {
    std::ostringstream output_filename;
    output_filename << "layer" << std::setw(3) << std::setfill('0') << idx_layer
                    << ".output";
    dump_path1 = {path_helper_.GetFilePath(output_filename.str())};

    if (dumps_debug_files) {
      output_filename.str("");
      output_filename.clear();
      output_filename << "act_fp_" << std::setw(3) << std::setfill('0')
                      << idx_layer << ".txt";
      dump_path2 = {path_helper_.GetFilePath(output_filename.str())};
    }
  }

  std::ofstream dump_file1{dump_path1, std::ios::out | std::ios::app};
  if (!dump_file1.is_open()) {
    LOG(ERROR) << "Cannot open dump file : " << dump_path1;
    // TODO: handle error
  }

  std::ofstream dump_file2;
  if (dumps_debug_files) {
    if (dump_path2.has_value()) {
      dump_file2 = {dump_path2.value(), std::ios::out};
      if (!dump_file2.is_open()) {
        LOG(ERROR) << "Cannot open dump file : " << dump_path2.value();
        // TODO: handle error
      }
    } else {
      LOG(ERROR) << "Cannot create dump file path : act_fp_" << std::setw(3)
                 << std::setfill('0') << idx_layer << ".txt";
      // TODO: handle error
    }
  }

  const auto nn = output_tensor.n();
  const auto cc = output_tensor.c();
  const auto hh = output_tensor.h();
  const auto ww = output_tensor.w();
  const float *tensor_data = output_tensor.data<float>();

  auto get_element = [&](const float *image, int y, int x, int c) {
    const size_t idx = (hh * ww * c) + (ww * y) + x;
    return GetFp32Bits(image[idx]);
  };

  for (int b = 0; b < nn; ++b) {
    const auto *data = tensor_data + b * (hh * ww * cc);

    for (int y = 0; y < hh; ++y) {
      for (int x = 0; x < ww; ++x) {
        for (int c = 0; c < cc; ++c) {
          uint32_t v = get_element(data, y, x, c);
          dump_file1 << std::hex << std::uppercase << std::setw(8)
                     << std::setfill('0') << v << " ";
        }
      }
    }  // y,x,c
    dump_file1 << "\n";

    // assume n=1
    if (dumps_debug_files) {
      for (int c = 0; c < cc; ++c) {
        for (int y = 0; y < hh; ++y) {
          for (int x = 0; x < ww; ++x) {
            const size_t index = (hh * ww * c) + (ww * y) + x;
            dump_file2 << "output[" << index << "] = " << std::fixed
                       << std::setprecision(20) << data[index] << "\n";
          }
        }
      }  // y,x,z
    }
  }
}
