#ifndef PARSER_PARSER_HPP
#define PARSER_PARSER_HPP

#include <map>
#include <memory>
#include <string>

#include "enums/error.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "tl/expected.hpp"

namespace parser {
class Parser {
 public:
  tl::expected<void, SimulatorError> BuildNetwork(
      std::unique_ptr<Network> &network, const std::string &binary_path,
      const std::string &backend_type, bool do_quant);
  tl::expected<void, SimulatorError> DumpCalibratedModel(
      std::unique_ptr<Network> &network, const std::string &binary_path,
      const std::string &calibrated_model_dump_path);
  virtual ~Parser() {}

 protected:
  virtual tl::expected<void, SimulatorError> ReadGraphBinary(
      const std::string &binary_path) = 0;
  virtual tl::expected<void, SimulatorError> ParseGraphBinary(
      std::unique_ptr<Network> &) = 0;
  virtual tl::expected<void, SimulatorError> DumpGraphBinary(
      const std::string &binary_path) = 0;
  virtual tl::expected<void, SimulatorError> UpdateGraphThresholds(
      std::unique_ptr<Network> &network) = 0;
};
}  // namespace parser

#endif  // PARSER_PARSER_HPP
