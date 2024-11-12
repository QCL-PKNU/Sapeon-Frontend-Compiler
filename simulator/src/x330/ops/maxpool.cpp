#include "x330/ops/maxpool.hpp"

#include <memory>

#include "factory.hpp"
#include "inference_context.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "x330/x330_operation.hpp"

namespace x330 {

constexpr auto kOpsName = "Maxpool";

static bool kRegistered =
    Factory<X330Operation>::RegisterCreateFunction(kOpsName, Maxpool::Create);

std::unique_ptr<X330Operation> Maxpool::Create() {
  return std::make_unique<Maxpool>();
}

void Maxpool::PrepareQuantOperation(std::unique_ptr<Network>& network,
                                    const int idx_layer) {}

void Maxpool::Forward(Layer& layer, InferenceContext& ctx) {
  auto result = ForwardUnaryOperation(layer, ctx, kOpsName);
  if (!result) {
    ctx.SetOutputTensor(ctx.InputTensor(0));
  }
}
}  // namespace x330
