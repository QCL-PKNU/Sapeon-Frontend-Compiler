#ifndef BACKENDS_DELEGATE_QUANTIZATION_DUMP_HELPER_HPP
#define BACKENDS_DELEGATE_QUANTIZATION_DUMP_HELPER_HPP

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <tuple>

#include "arguments.hpp"
#include "enums/dump.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"

namespace quantization {
class QuantizationDumpHelper {
 public:
  QuantizationDumpHelper(Arguments& args);
  void DumpX220QuantizedNetworkInfo(std::unique_ptr<Network>& network);
  void DumpX330LayerFilter(Layer& layer);
  void DumpX330LayerFilterFP(Layer& layer);
  void DumpX330UpdatedEbiases(
      Layer& layer, const std::tuple<int, int, int>& ebiases,
      const std::optional<std::filesystem::path>& updated_ebias_dump_path,
      bool truncates_file = false);

 private:
  DumpLevel dump_level_;
  std::optional<std::filesystem::path> dump_dir_;
  std::string model_file_name_;
};
}  // namespace quantization
#endif  // BACKENDS_DELEGATE_QUANTIZATION_DUMP_HELPER_HPP
