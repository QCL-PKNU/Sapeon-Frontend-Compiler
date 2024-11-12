#include "x330/ops/avgpool.hpp"

#include <memory>

#include "factory.hpp"
#include "inference_context.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "x330/x330_operation.hpp"

namespace x330 {

constexpr auto kOpsName = "Avgpool";

static bool kRegistered =
    Factory<X330Operation>::RegisterCreateFunction(kOpsName, Avgpool::Create);

std::unique_ptr<X330Operation> Avgpool::Create() {
  return std::make_unique<Avgpool>();
}

void Avgpool::PrepareQuantOperation(std::unique_ptr<Network>& network,
                                    const int idx_layer) {}

void Avgpool::Forward(Layer& layer, InferenceContext& ctx) {
  auto result = ForwardUnaryOperation(layer, ctx, kOpsName);
  if (!result) {
    ctx.SetOutputTensor(ctx.InputTensor(0));
  }
}
}  // namespace x330
