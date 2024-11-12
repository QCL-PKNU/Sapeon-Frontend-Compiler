#include "x330/ops/route.hpp"

#include <memory>

#include "factory.hpp"
#include "inference_context.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "operations/cpu_operation.hpp"
#include "x330/x330_operation.hpp"

namespace x330 {

constexpr auto kOpsName = "Route";

static bool kRegistered =
    Factory<X330Operation>::RegisterCreateFunction(kOpsName, Route::Create);

std::unique_ptr<X330Operation> Route::Create() {
  return std::make_unique<Route>();
}

void Route::PrepareQuantOperation(std::unique_ptr<Network>& network,
                                  const int idx_layer) {}

void Route::Forward(Layer& layer, InferenceContext& ctx) {
  auto& cfg = layer.x330_quant_config();
  cfg.num_samples++;

  auto num_inputs = layer.predecessors().size();

  for (int i = 0; i < num_inputs; i++) {
    ConvertInputTensor(ctx.InputTensor(i), cfg);
  }

  auto route = Factory<CpuOperation>::CreateInstance(kOpsName);
  route->Forward(layer, ctx);

  auto output = ctx.OutputTensor();

  ConvertOutputTensor(output, cfg);
}
}  // namespace x330
