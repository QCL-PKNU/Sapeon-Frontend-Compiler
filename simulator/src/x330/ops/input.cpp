#include "x330/ops/input.hpp"

#include <cstring>
#include <memory>

#include "factory.hpp"
#include "inference_context.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"

namespace x330 {

static bool kRegistered =
    Factory<X330Operation>::RegisterCreateFunction("Input", Input::Create);

std::unique_ptr<X330Operation> Input::Create() {
  return std::make_unique<Input>();
}

void Input::PrepareQuantOperation(std::unique_ptr<Network>& network,
                                  const int idx_layer) {}

void Input::Forward(Layer& layer, InferenceContext& ctx) {
  auto& cfg = layer.x330_quant_config();
  cfg.num_samples++;

  auto input = ctx.InputTensor(0);

  ConvertInputTensor(input, cfg);

  auto output = std::make_shared<Tensor>(input->n(), input->c(), input->h(),
                                         input->w(), input->dtype());
  std::memcpy(output->data(), input->data(), input->size());

  ConvertOutputTensor(output, cfg);

  ctx.SetOutputTensor(output);
}
}  // namespace x330
