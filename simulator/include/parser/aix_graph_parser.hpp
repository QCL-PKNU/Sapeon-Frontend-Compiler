#ifndef PARSER_AIX_GRAPH_PARSER_HPP
#define PARSER_AIX_GRAPH_PARSER_HPP

#include <aixh.pb.h>
using aixh::AIXGraph;
using aixh::AIXLayer;
using AIXTensor = aixh::AIXLayer_AIXTensor;
using AIXConvolutionDesc = aixh::AIXLayer_AIXConvolutionDesc;
using AIXEWAddDesc = aixh::AIXLayer_AIXEWAddDesc;
using AIXSamplingDesc = aixh::AIXLayer_AIXSamplingDesc;

#include <map>
#include <memory>
#include <string>

#include "enums/error.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "parser/parser.hpp"
#include "tl/expected.hpp"

namespace parser {
class AIXGraphParser : public Parser {
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
  Layer ParseAIXLayer(const AIXLayer &aix_layer);
  std::shared_ptr<Tensor> ParseAIXTensor(const AIXTensor &aix_tensor,
                                         int groups);
  std::shared_ptr<Descriptor> ParseAIXConvolutionDesc(
      const AIXConvolutionDesc &convdesc);
  std::shared_ptr<Descriptor> ParseAIXEWAddDesc(const AIXEWAddDesc &ewadddesc);
  std::shared_ptr<Descriptor> ParseAIXSamplingDesc(
      const AIXSamplingDesc &samplingdesc);
  AIXGraph aix_graph_;
};
}  // namespace parser

#endif  // PARSER_AIX_GRAPH_PARSER_HPP
