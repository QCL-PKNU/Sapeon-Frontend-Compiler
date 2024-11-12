#ifndef BACKENDS_DELEGATE_X330_QUANTIZATION_DELEGATE_HPP
#define BACKENDS_DELEGATE_X330_QUANTIZATION_DELEGATE_HPP

#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <string>

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "backends/delegate/quantization_delegate.hpp"
#include "backends/delegate/quantization_dump_helper.hpp"
#include "enums/error.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "tl/expected.hpp"

namespace quantization {
class X330QuantizationDelegate : public QuantizationDelegate {
 public:
  X330QuantizationDelegate(Backend &parent, Arguments &args);
  tl::expected<void, SimulatorError> Quantize(
      std::unique_ptr<Network> &network) override;
  virtual ~X330QuantizationDelegate() {}

 private:
  tl::expected<void, SimulatorError> ParseConfigFile(
      const std::filesystem::path &file_path,
      std::unique_ptr<Network> &network);
  void InitDefaultQuantConfig(std::unique_ptr<Network> &network);
  tl::expected<void, SimulatorError> PrepareQuantOperation(
      std::unique_ptr<Network> &network);
  std::optional<std::filesystem::path> cfg_file_path_;
  std::optional<std::filesystem::path> max_file_path_;
  std::optional<std::filesystem::path> updated_ebias_dump_file_path_;
  QuantizationDumpHelper dump_;
  Backend &parent_;
};
}  // namespace quantization

#endif  // BACKENDS_DELEGATE_X330_QUANTIZATION_DELEGATE_HPP
