#ifndef PARSER_SPEAR_GRAPH_PARSER_HPP
#define PARSER_SPEAR_GRAPH_PARSER_HPP

#include <any>
#include <map>
#include <memory>
#include <string>

#include "enums/error.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "parser/parser.hpp"
#include "spear.proto.e8e8.pb.h"
#include "tl/expected.hpp"

namespace parser {
class SpearGraphParser : public Parser {
 public:
  static std::unique_ptr<Parser> CreateParser();

 private:
  tl::expected<void, SimulatorError> ReadGraphBinary(
      const std::string &binary_path) override;
  tl::expected<void, SimulatorError> ParseGraphBinary(
      std::unique_ptr<Network> &network) override;
  tl::expected<void, SimulatorError> DumpGraphBinary(
      const std::string &binary_path) override;
  tl::expected<void, SimulatorError> UpdateGraphThresholds(
      std::unique_ptr<Network> &network) override;
  tl::expected<Layer, SimulatorError> ParseSPLayer(
      const sapeon::simulator::SPLayer &spear_layer);
  tl::expected<std::shared_ptr<Tensor>, SimulatorError> ParseSPTensor(
      const sapeon::simulator::SPLayer_SPTensor &spear_tensor, int groups);
  tl::expected<std::shared_ptr<Descriptor>, SimulatorError>
  ParseSPConvolutionDesc(
      const sapeon::simulator::SPLayer_SPConvolutionDesc &convdesc,
      const sapeon::simulator::SPLayer &layer);
  tl::expected<std::shared_ptr<Descriptor>, SimulatorError> ParseSPEWAddDesc(
      const sapeon::simulator::SPLayer_SPEWAddDesc &ewadddesc,
      const sapeon::simulator::SPLayer &layer);
  tl::expected<std::shared_ptr<Descriptor>, SimulatorError> ParseSPEWMulDesc(
      const sapeon::simulator::SPLayer_SPEWMulDesc &ewmuldesc,
      const sapeon::simulator::SPLayer &layer);
  tl::expected<std::shared_ptr<Descriptor>, SimulatorError> ParseSPSamplingDesc(
      const sapeon::simulator::SPLayer_SPSamplingDesc &samplingdesc,
      const sapeon::simulator::SPLayer &layer);
  tl::expected<std::any, SimulatorError> ParseSPAttribute(
      const std::string &name, const sapeon::simulator::SPLayer &layer);
  sapeon::simulator::SPGraph spear_graph_;
};
}  // namespace parser

#endif  // PARSER_SPEAR_GRAPH_PARSER_HPP
