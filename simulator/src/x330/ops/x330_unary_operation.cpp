#include "x330/ops/x330_unary_operation.hpp"

#include <memory>

#include "enums/error.hpp"
#include "factory.hpp"
#include "inference_context.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "operations/cpu_operation.hpp"
#include "tl/expected.hpp"
#include "x330/x330_operation.hpp"

namespace x330 {

tl::expected<void, SimulatorError> X330UnaryOperation::ForwardUnaryOperation(
    Layer& layer, InferenceContext& ctx, const std::string& ops_name) {
  auto& cfg = layer.x330_quant_config();
  cfg.num_samples++;

  auto op = Factory<CpuOperation>::CreateInstance(ops_name);
  if (op == nullptr) {
    DLOG(ERROR) << "Failed to create CpuOperation: " << ops_name;
    return tl::make_unexpected(SimulatorError::kCreateInstanceError);
  }

  auto input = ctx.InputTensor(0);
  ConvertInputTensor(input, cfg);

  op->Forward(layer, ctx);

  auto output = ctx.OutputTensor();
  ConvertOutputTensor(output, cfg);

  return {};
}
}  // namespace x330
