#include "x330/ops/upsample.hpp"

#include <memory>

#include "factory.hpp"
#include "inference_context.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "x330/x330_operation.hpp"

namespace x330 {

constexpr auto kOpsName = "Upsample";

static bool kRegistered =
    Factory<X330Operation>::RegisterCreateFunction(kOpsName, Upsample::Create);

std::unique_ptr<X330Operation> Upsample::Create() {
  return std::make_unique<Upsample>();
}

void Upsample::PrepareQuantOperation(std::unique_ptr<Network>& network,
                                     const int idx_layer) {}

void Upsample::Forward(Layer& layer, InferenceContext& ctx) {
  auto result = ForwardUnaryOperation(layer, ctx, kOpsName);
  if (!result) {
    ctx.SetOutputTensor(ctx.InputTensor(0));
  }
}
}  // namespace x330
