#include "x330/ops/pixelshuffle.hpp"

#include <memory>

#include "factory.hpp"
#include "inference_context.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "x330/x330_operation.hpp"

namespace x330 {

constexpr auto kOpsName = "Pixelshuffle";

static bool kRegistered = Factory<X330Operation>::RegisterCreateFunction(
    kOpsName, Pixelshuffle::Create);

std::unique_ptr<X330Operation> Pixelshuffle::Create() {
  return std::make_unique<Pixelshuffle>();
}

void Pixelshuffle::PrepareQuantOperation(std::unique_ptr<Network>& network,
                                         const int idx_layer) {}

void Pixelshuffle::Forward(Layer& layer, InferenceContext& ctx) {
  auto result = ForwardUnaryOperation(layer, ctx, kOpsName);
  if (!result) {
    ctx.SetOutputTensor(ctx.InputTensor(0));
  }
}
}  // namespace x330
