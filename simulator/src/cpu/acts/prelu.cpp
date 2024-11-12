#include "cpu/acts/prelu.hpp"

#include "factory.hpp"

#define BASE CpuOperation
#define NAME PReLU
#define CLASS cpu::NAME
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

using std::make_unique;
using std::unique_ptr;
using x220::QuantConfig;

static bool kRegistered =
    Factory<BASE>::RegisterCreateFunction(GET_STR(NAME), CLASS::Create);

std::unique_ptr<BASE> SCOPE::Create() { return std::make_unique<CLASS>(); }

bool SCOPE::CheckValidOperation(Layer& layer, Dimension input_dimension) {
  if (layer.negative_slope().empty()) {
    LOG(FATAL) << "negative slope not found\n";
    return false;
  }
  return true;
};

Dimension SCOPE::CalculateOutputDimension(Layer& layer,
                                          Dimension input_dimension) {
  return input_dimension;
}

void SCOPE::ActivationForward(Layer& layer) {
  const auto itype = input_->dtype();
  const auto otype = output_->dtype();
  if (layer.negative_slope().empty()) {
    LOG(FATAL) << "negative slope not found\n";
  }
  auto activation = [neg_slope =
                         layer.negative_slope(0)](auto x) -> decltype(x) {
    return (x >= 0) ? x * 1.0f : x * neg_slope;
  };

  if (itype == dty::DataType::FP32 && otype == dty::DataType::FP32) {
    Activation::ActivationForward<float, float>(std::move(activation));
  } else if (itype == dty::DataType::FP64 && otype == dty::DataType::FP64) {
    Activation::ActivationForward<double, double>(std::move(activation));
  } else {
    LOG(ERROR) << "Not implemented forward type: itype=" << dty::NameOf(itype)
               << ", otype=" << dty::NameOf(otype) << '\n';
    exit(1);
  }
}

#ifndef CONFIDENTIAL_FEATURES
void SCOPE::ActivationQuantForward(Layer& layer) {
  LOG(ERROR) << "Not implemented forward type" << '\n';
  exit(1);
}
#endif
