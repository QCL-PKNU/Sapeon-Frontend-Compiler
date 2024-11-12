#include "parser/parser.hpp"

#define CLASS Parser
#define SCOPE CLASS

#include <cassert>
#include <map>
using std::map;
#include <memory>
using std::unique_ptr;
#include <string>
using std::string;

#include "glog/logging.h"
#include "tl/expected.hpp"
using tl::expected;
using tl::make_unexpected;

#include "enums/error.hpp"
#include "factory.hpp"
#include "network/layer.hpp"
#include "operations/cpu_operation.hpp"
#ifdef GPU
#include "operations/cudnn_operation.hpp"
#endif

namespace parser {
expected<void, SimulatorError> SCOPE::BuildNetwork(unique_ptr<Network> &network,
                                                   const string &binary_path,
                                                   const string &backend_type,
                                                   bool do_quant) {
  auto result = ReadGraphBinary(binary_path);
  if (!result) {
    DLOG(ERROR) << "ReadGraphBinary failed";
    return make_unexpected(result.error());
  }

  result = ParseGraphBinary(network);
  if (!result) {
    DLOG(ERROR) << "ParseGraphBinary failed";
    return make_unexpected(result.error());
  }


  LOG(INFO) << "---------------------------------------------- Parsing Binary Graph Successfully";

  bool valid_graph = network->CheckValidNetwork(backend_type, do_quant);
  
  if (!valid_graph) {
    DLOG(ERROR) << "CheckValidNetwork failed";
    return make_unexpected(SimulatorError::kInvalidModel);
  }


  LOG(INFO) << "---------------------------------------------- Network is validated Successfully with status: " << valid_graph;

  return {};
}

expected<void, SimulatorError> SCOPE::DumpCalibratedModel(
    unique_ptr<Network> &network, const string &binary_path,
    const string &calibrated_model_dump_path) {
  auto result = ReadGraphBinary(binary_path);
  if (!result) return make_unexpected(result.error());

  result = UpdateGraphThresholds(network);
  if (!result) return make_unexpected(result.error());

  result = DumpGraphBinary(calibrated_model_dump_path);
  if (!result) return make_unexpected(result.error());

  return {};
}

}  // namespace parser
