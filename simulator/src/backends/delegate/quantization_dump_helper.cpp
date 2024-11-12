#include "backends/delegate/quantization_dump_helper.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>

#include "arguments.hpp"
#include "datatype.hpp"
#include "enums/dump.hpp"
#include "glog/logging.h"
#include "network/tensor.hpp"

namespace fs = std::filesystem;

namespace quantization {
QuantizationDumpHelper::QuantizationDumpHelper(Arguments& args) {
  dump_level_ = GetDumpLevel(args.dump_level());
  dump_dir_ = args.dump_dir();
  model_file_name_ = fs::path{args.model_path()}.filename().string();
}

void QuantizationDumpHelper::DumpX220QuantizedNetworkInfo(
    std::unique_ptr<Network>& network) {
  if (dump_level_ == DumpLevel::DUMP_NONE) {
    LOG(INFO) << "dump-level is none. Does not dump quantized network info.";
    return;
  }
  const auto dump_path =
      dump_dir_.value() / fs::path{model_file_name_ + "_log"};
  std::ofstream dump_file(dump_path);

  dump_file << "layer grp       layer name       input(HWC)    output(HWC)     "
               "     dtypes(i,o,f) cluster   infus  outfus tilecnt   iblkh   "
               "oblkh filter(HWN,Sx/Sy)   GFLOPs\n";

  for (auto& layer : network->layers()) {
    auto id = layer.id();
    int grp = 1;
    auto& layer_name = layer.type();
    auto in_dtype =
        layer.x220_quant_config().in_dtype() == x220::DataType::DTY_SINT8
            ? std::string{"sint8"}
            : std::string{"uint8"};
    auto out_dtype =
        layer.x220_quant_config().out_dtype() == x220::DataType::DTY_SINT8
            ? std::string{"sint8"}
            : std::string{"uint8"};

    auto filter_dtype = std::string{"sint8"};

    auto& input = layer.input_dimensions(0);
    std::stringstream input_str;
    input_str << std::setw(5) << input.h() << "x" << std::setw(4) << input.w()
              << "x" << std::setw(4) << input.c();

    auto& output = layer.output_dimension();
    std::stringstream output_str;
    output_str << std::setw(5) << output.h() << "x" << std::setw(4)
               << output.w() << "x" << std::setw(4) << output.c();

    std::stringstream filter_str;
    if (layer.HasConvolutionDescriptor()) {
      auto convdesc = layer.convolution();
      grp = convdesc->groups();
      auto& filter = layer.filter()->dimension();
      filter_str << std::setw(4) << filter.h() << "x" << std::setw(2)
                 << filter.w() << "x" << std::setw(4) << filter.n() << ","
                 << std::setw(2) << convdesc->stride_width() << "/"
                 << std::setw(2) << convdesc->stride_height();
    } else if (layer.HasSamplingDescriptor()) {
      auto sampldesc = layer.sampling();
      filter_str << std::setw(4) << sampldesc->window_height() << "x"
                 << std::setw(2) << sampldesc->window_width() << "x"
                 << std::setw(4) << "    "
                 << "," << std::setw(2) << sampldesc->stride_width() << "/"
                 << std::setw(2) << sampldesc->stride_height();
    } else {
      filter_str << "                  ";
    }

    dump_file << std::setw(5) << id << std::setw(4) << grp << std::setw(17)
              << layer_name << std::setw(15) << input_str.str() << "->"
              << std::setw(15) << output_str.str() << std::setw(7) << in_dtype
              << "," << std::setw(7) << out_dtype << "," << std::setw(7)
              << filter_dtype
              << "       1       1       1       1       1       1"
              << std::setw(18) << filter_str.str() << "    0.000\n";
  }
  dump_file.close();
}

namespace {
auto GetFp32Bits(float fp32) {
  union {
    float f;
    uint32_t u;
  } tmp = {fp32};
  return tmp.u;
};

auto GetDumpPath(const std::optional<std::string>& dump_dir,
                 const std::string& filename) {
  fs::path dir{"."};
  dir /= fs::path{dump_dir.value_or("dump")};
  dir /= fs::path{filename};
  return dir;
}
}  // namespace

void QuantizationDumpHelper::DumpX330LayerFilter(Layer& layer) {
  // if (!aix_dump_data_enabled()) return;
  if (dump_level_ == DumpLevel::DUMP_NONE) return;
  const bool dumps_debug_files = dump_level_ == DumpLevel::DUMP_DEBUG;

  fs::path dump_path1;
  std::optional<fs::path> dump_path2{std::nullopt};
  std::optional<fs::path> dump_path3{std::nullopt};

  {
    std::ostringstream output_filename;
    const auto idx_layer = layer.id();
    output_filename << "layer" << std::setw(3) << std::setfill('0') << idx_layer
                    << ".filter";
    dump_path1 = {GetDumpPath(dump_dir_, output_filename.str())};

    output_filename.str("");
    output_filename.clear();

    output_filename << "nf_weight_" << std::setw(3) << std::setfill('0')
                    << idx_layer << ".txt";
    dump_path2 = {GetDumpPath(dump_dir_, output_filename.str())};

    output_filename.str("");
    output_filename.clear();

    output_filename << "nf_bias_" << std::setw(3) << std::setfill('0')
                    << idx_layer << ".txt";
    dump_path3 = {GetDumpPath(dump_dir_, output_filename.str())};
  }

  std::ofstream dump_file1;
  std::ofstream dump_file2;
  std::ofstream dump_file3;
  {
    dump_file1 = {dump_path1, std::ios::out};
    if (!dump_file1.is_open()) {
      LOG(ERROR) << "Cannot open dump file : " << dump_path1;
    }
    if (dumps_debug_files) {
      const auto idx_layer = layer.id();
      if (dump_path2.has_value()) {
        dump_file2 = {dump_path2.value(), std::ios::out};
        if (!dump_file2.is_open()) {
          LOG(ERROR) << "Cannot open dump file : " << dump_path2.value();
          // TODO: handle error
        }
      } else {
        LOG(ERROR) << "Cannot create dump file path : "
                   << (dump_dir_.has_value() ? dump_dir_.value() : "dump")
                   << "/nf_weight_" << std::setw(3) << std::setfill('0')
                   << idx_layer << ".txt";
        // TODO: handle error
      }

      if (dump_path3.has_value()) {
        dump_file3 = {dump_path3.value(), std::ios::out};
        if (!dump_file3.is_open()) {
          LOG(ERROR) << "Cannot open dump file : " << dump_path3.value();
          // TODO: handle error
        }
      } else {
        LOG(ERROR) << "Cannot create dump file path : "
                   << (dump_dir_.has_value() ? dump_dir_.value() : "dump")
                   << "/nf_weight_" << std::setw(3) << std::setfill('0')
                   << idx_layer << ".txt";
        // TODO: handle error
      }
    }
  }

  const auto filter = layer.filter();
  const auto size_w = filter->w();
  const auto size_h = filter->h();
  const auto in_c = filter->c();
  const auto out_c = filter->n();
  const float* filter_data = filter->data<float>();

  auto get_weight = [&](const float* data, int y, int x, int c) {
    const size_t idx = (size_w * size_h * c) + (size_w * y) + x;
    return GetFp32Bits(data[idx]);
  };

  for (int f = 0; f < out_c; ++f) {
    // Weight
    auto weights = filter_data + f * (size_w * size_h * in_c);

    for (int y = 0; y < size_h; ++y) {
      for (int x = 0; x < size_w; ++x) {
        for (int c = 0; c < in_c; ++c) {
          uint32_t v = get_weight(weights, y, x, c);
          dump_file1 << std::hex << std::uppercase << std::setw(8)
                     << std::setfill('0') << v << " ";
        }
      }
    }  // y,x,c

    // Bias
    {
      uint32_t v = GetFp32Bits(layer.bias()->data<float>()[f]);
      dump_file1 << std::hex << std::uppercase << std::setw(8)
                 << std::setfill('0') << v << " ";
    }

    // Scale
    {
      float pos_scale = 1.0;
      float neg_scale = 1.0;
      const auto& activation_type = layer.activation_type();
      if (activation_type == "LeakyReLU") {
        neg_scale = 0.1;
      } else if (activation_type == "PReLU") {
        if (layer.negative_slope().empty()) {
          // TODO: handle error
        }
        neg_scale = layer.negative_slope(0);
        // Need to implement cwReLU?
      } else if (activation_type == "ReLU") {
        neg_scale = 0.0;
      }

      const uint32_t v0 = GetFp32Bits(pos_scale);
      const uint32_t v1 = GetFp32Bits(neg_scale);
      dump_file1 << std::hex << std::uppercase << std::setw(8)
                 << std::setfill('0') << v0 << " " << std::hex << std::uppercase
                 << std::setw(8) << std::setfill('0') << v1;
    }
    dump_file1 << "\n";

    // FIXME(jieun): need to check nf_weight vs. fp_weight
    if (dumps_debug_files) {
      // weight
      for (int c = 0; c < in_c; ++c) {
        for (int y = 0; y < size_h; ++y) {
          for (int x = 0; x < size_w; ++x) {
            const size_t index = (size_w * size_h * c) + (size_w * y) + x;
            dump_file2 << "output[" << (size_w * size_h * in_c * f) + index
                       << "] = " << std::fixed << std::setprecision(40)
                       << weights[index] << "\n";
          }
        }
      }

      // bias
      dump_file3 << "output[" << f << "] = " << std::fixed
                 << std::setprecision(40) << layer.bias()->data<float>()[f]
                 << "\n";
    }
  }  // filter
}

void QuantizationDumpHelper::DumpX330LayerFilterFP(Layer& layer) {
  if (dump_level_ != DumpLevel::DUMP_DEBUG) return;

  fs::path dump_path1;
  fs::path dump_path2;
  {
    std::ostringstream output_filename;
    const auto idx_layer = layer.id();
    output_filename << "fp_weight_" << std::setw(3) << std::setfill('0')
                    << idx_layer << ".txt";
    dump_path1 = {GetDumpPath(dump_dir_, output_filename.str())};

    output_filename.str("");
    output_filename.clear();

    output_filename << "fp_bias_" << std::setw(3) << std::setfill('0')
                    << idx_layer << ".txt";
    dump_path2 = {GetDumpPath(dump_dir_, output_filename.str())};
  }

  std::ofstream dump_file1;
  std::ofstream dump_file2;
  {
    dump_file1 = {dump_path1, std::ios::out};
    if (!dump_file1.is_open()) {
      LOG(ERROR) << "Cannot open dump file : " << dump_path1;
    }

    dump_file2 = {dump_path2, std::ios::out};
    if (!dump_file2.is_open()) {
      LOG(ERROR) << "Cannot open dump file : " << dump_path1;
    }
  }

  const auto filter = layer.filter();
  const auto size_w = filter->w();
  const auto size_h = filter->h();
  const auto in_c = filter->c();
  const auto out_c = filter->n();
  const float* filter_data = filter->data<float>();

  const auto bias = layer.bias();
  const float* bias_data = bias->data<float>();

  for (int n = 0; n < out_c; ++n) {
    // weights
    for (int c = 0; c < in_c; ++c) {
      for (int y = 0; y < size_h; ++y) {
        for (int x = 0; x < size_w; ++x) {
          const size_t index = (size_w * size_h * in_c * n) +
                               (size_w * size_h * c) + (size_w * y) + x;
          dump_file1 << "output[" << index << "] = " << std::fixed
                     << std::setprecision(40) << filter_data[index] << "\n";
        }
      }
    }
    // bias
    dump_file2 << "output[" << n << "] = " << std::fixed
               << std::setprecision(50) << bias_data[n] << "\n";
  }
}

void QuantizationDumpHelper::DumpX330UpdatedEbiases(
    Layer& layer, const std::tuple<int, int, int>& ebiases,
    const std::optional<fs::path>& updated_ebias_dump_path,
    bool truncates_file) {
  if (dump_level_ == DumpLevel::DUMP_NONE) return;
  const auto& [prev_in_ebias, prev_actin_ebias, prev_out_ebias] = ebiases;
  const auto updated_in_ebias = layer.x330_quant_config().input_ebias;
  const auto updated_actin_ebias = layer.x330_quant_config().actin_ebias;
  const auto updated_out_ebias = layer.x330_quant_config().output_ebias;

  const auto idx_layer = layer.id();
  fs::path dump_path =
      updated_ebias_dump_path.value_or(GetDumpPath(dump_dir_, "updated.ebias"));

  std::ofstream dump_file;

  if (truncates_file) {
    dump_file.open(dump_path, std::ios::out);
    dump_file << "#   |   PREV EBIAS  | UPDATED EBIAS |\n";
    dump_file << "# L |   I   A   O   |   I   A   O   |\n";
  } else {
    dump_file.open(dump_path, std::ios::app);
  }
  std::stringstream output;
  output << std::setw(4) << idx_layer << "  ";
  output << std::setw(3) << prev_in_ebias << " ";
  output << std::setw(3) << prev_actin_ebias << " ";
  output << std::setw(3) << prev_out_ebias << "     ";
  output << std::setw(3) << updated_in_ebias << " ";
  output << std::setw(3) << updated_actin_ebias << " ";
  output << std::setw(3) << updated_out_ebias << "\n";
  dump_file << output.str();
}

}  // namespace quantization
