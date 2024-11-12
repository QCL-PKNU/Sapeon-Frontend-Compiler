#include "x330/ops/activations.hpp"

#include <memory>

#include "factory.hpp"
#include "glog/logging.h"
#include "inference_context.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "operations/cpu_operation.hpp"
#include "x330/x330_operation.hpp"

namespace x330 {

static bool kRegistered = Factory<X330Operation>::RegisterCreateFunction(
    "Activations", Activations::Create);

std::unique_ptr<X330Operation> Activations::Create() {
  return std::make_unique<Activations>();
}

void Activations::PrepareQuantOperation(std::unique_ptr<Network>& network,
                                        const int idx_layer) {}

void Activations::Forward(Layer& layer, InferenceContext& ctx) {
  auto& cfg = layer.x330_quant_config();

  if (layer.operation_types().at(0) == "Activations") {
    cfg.num_samples++;
    auto input = ctx.InputTensor(0);
    ConvertInputTensor(input, cfg);
  }

  const auto& act_name = layer.activation_type();
  if (cfg.actfn_lut && !(act_name == "Identity" && act_name == "LeakyReLU")) {
    // Do LUT Activation
    auto activation = Factory<X330Operation>::CreateInstance(act_name);
    if (activation == nullptr) {
      LOG(ERROR) << "Invalid activation: " << act_name;
      // TODO: add error handling logics
    }
    activation->Forward(layer, ctx);
  } else {
    auto activation = Factory<CpuOperation>::CreateInstance(act_name);
    if (activation == nullptr) {
      LOG(ERROR) << "Invalid activation: " << act_name;
      // TODO: add error handling logics
    }
    activation->Forward(layer, ctx);
  }

  auto output = ctx.OutputTensor();
  ConvertOutputTensor(output, cfg);
}

}  // namespace x330
