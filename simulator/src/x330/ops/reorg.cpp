#include "x330/ops/reorg.hpp"

#include <memory>

#include "factory.hpp"
#include "inference_context.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "x330/x330_operation.hpp"

namespace x330 {

constexpr auto kOpsName = "Reorg";

static bool kRegistered =
    Factory<X330Operation>::RegisterCreateFunction(kOpsName, Reorg::Create);

std::unique_ptr<X330Operation> Reorg::Create() {
  return std::make_unique<Reorg>();
}

void Reorg::PrepareQuantOperation(std::unique_ptr<Network>& network,
                                  const int idx_layer) {}

void Reorg::Forward(Layer& layer, InferenceContext& ctx) {
  auto result = ForwardUnaryOperation(layer, ctx, kOpsName);
  if (!result) {
    ctx.SetOutputTensor(ctx.InputTensor(0));
  }
}
}  // namespace x330
