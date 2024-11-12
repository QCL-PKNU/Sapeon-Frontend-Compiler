#include "x330/ops/element_wise_add.hpp"

#include <algorithm>
#include <memory>

#include "factory.hpp"
#include "inference_context.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "operations/cpu_operation.hpp"
#include "x330/x330_operation.hpp"

namespace x330 {

constexpr auto kOpsName = "EWAdd";

static bool kRegistered =
    Factory<X330Operation>::RegisterCreateFunction(kOpsName, EWAdd::Create);

std::unique_ptr<X330Operation> EWAdd::Create() {
  return std::make_unique<EWAdd>();
}
void EWAdd::PrepareQuantOperation(std::unique_ptr<Network>& network,
                                  const int idx_layer) {}

void EWAdd::Forward(Layer& layer, InferenceContext& ctx) {
  auto& cfg = layer.x330_quant_config();
  cfg.num_samples++;

  auto num_inputs = layer.predecessors().size();

  for (int i = 0; i < num_inputs; i++) {
    ConvertInputTensor(ctx.InputTensor(i), cfg);
  }

  auto ewadd = Factory<CpuOperation>::CreateInstance(kOpsName);
  ewadd->Forward(layer, ctx);

  auto output = ctx.OutputTensor();

  ConvertActInTensor(output, cfg);

  const auto& op_types = layer.operation_types();
  const bool activation_not_exists = std::find(op_types.begin(), op_types.end(),
                                               "Activations") == op_types.end();
  if (activation_not_exists) {
    ConvertOutputTensor(output, cfg);
  }
}
}  // namespace x330
