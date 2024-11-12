#pragma once

#include <memory>
#include <string>
#include <vector>

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "backends/delegate/collect_delegate.hpp"
#include "enums/dump.hpp"
#include "enums/error.hpp"
#include "network/network.hpp"
#include "tl/expected.hpp"

namespace collect {
class X330CollectDelegate final : public CollectDelegate {
 public:
  X330CollectDelegate(Backend &parent, Arguments &args);
  tl::expected<void, SimulatorError> Collect(
      std::unique_ptr<Network> &network) override;

 private:
  void InitFPQuantConfig(std::unique_ptr<Network> &network);
  tl::expected<void, SimulatorError> CollectMaxes(
      std::unique_ptr<Network> &network);
  tl::expected<void, SimulatorError> DumpQuantMax(
      std::unique_ptr<Network> &network, const std::string &file_path,
      DumpLevel level);
  void TruncateX330LayerMax(const std::string &file_path);
  void DumpX330LayerMax(Layer &layer, const std::string &file_path,
                        DumpLevel level);
  Backend &parent_;
  std::string quant_max_path_;
  std::vector<std::string> collect_image_paths_;
  DumpLevel dump_level_;
};
}  // namespace collect
