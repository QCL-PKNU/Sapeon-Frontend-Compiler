#include "cpu/acts/sigmoid.hpp"

#define BASE CpuOperation
#define NAME Sigmoid
#define CLASS cpu::NAME
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

using std::make_unique;
using std::unique_ptr;
using x220::QuantConfig;

#include "datatype.hpp"
using dty::DataType;
#include "factory.hpp"
#include "glog/logging.h"
#include "network/dimension.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cpu_operation.hpp"
#include "utility.hpp"

static bool kRegistered =
    Factory<BASE>::RegisterCreateFunction(GET_STR(NAME), CLASS::Create);

std::unique_ptr<BASE> SCOPE::Create() { return std::make_unique<CLASS>(); }

bool SCOPE::CheckValidOperation(Layer &layer, Dimension input_dimension) {
  return true;
}

Dimension SCOPE::CalculateOutputDimension(Layer &layer,
                                          Dimension input_dimension) {
  return input_dimension;
}

void SCOPE::ActivationForward(Layer &layer) {
  const auto itype = input_->dtype();
  const auto otype = output_->dtype();
  auto activation = [](auto x) -> decltype(x) { return 1.f / (1.f + exp(-x)); };

  if (itype == dty::DataType::FP32 && otype == dty::DataType::FP32) {
    Activation::ActivationForward<float, float>(std::move(activation));
  } else if (itype == dty::DataType::FP64 && otype == dty::DataType::FP64) {
    Activation::ActivationForward<double, double>(std::move(activation));
  } else {
    DLOG(FATAL) << "Not implemented forward type: itype=" << dty::NameOf(itype)
                << ", otype=" << dty::NameOf(otype) << '\n';
  }
}

#ifndef CONFIDENTIAL_FEATURES
void SCOPE::ActivationQuantForward(Layer &layer) {
  LOG(ERROR) << "Not implemented forward type" << '\n';
  exit(1);
}
#endif
