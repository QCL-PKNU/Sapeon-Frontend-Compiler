#include "cpu/ops/activations.hpp"

#define BASE CpuOperation
#define NAME Activations
#define CLASS cpu::NAME
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

#include <memory>
#include <string>

#include "datatype.hpp"
#include "factory.hpp"
#include "glog/logging.h"
#include "network/dimension.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cpu_operation.hpp"

static bool kRegistered =
    Factory<BASE>::RegisterCreateFunction(GET_STR(NAME), CLASS::Create);

std::unique_ptr<BASE> SCOPE::Create() { return std::make_unique<CLASS>(); }

void SCOPE::Forward(Layer &layer, InferenceContext &ctx) {
  const auto &name = layer.activation_type();
  auto p_act = Factory<CpuOperation>::CreateInstance(name);

  p_act->Forward(layer, ctx);
}

bool SCOPE::CheckValidOperation(Layer &layer, Dimension input_dimension) {
  const auto &name = layer.activation_type();
  auto p_activation = Factory<CpuOperation>::CreateInstance(name);
  if (p_activation == nullptr) {
    LOG(ERROR) << "Invalid activation: " << name;
    return false;
  }
  return true;
}

Dimension SCOPE::CalculateOutputDimension(Layer &layer,
                                          Dimension input_dimension) {
  return input_dimension;
}
