#include "cpu/acts/relu6.hpp"

#define BASE CpuOperation
#define NAME ReLU6
#define CLASS cpu::NAME
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

using dty::DataType;
using std::make_shared;
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

void SCOPE::InitOutputTensor(dty::DataType dtype) {
  DataType out_dtype;
  switch (dtype) {
    case dty::DataType::SINT8:
      out_dtype = dty::DataType::UINT8;
      break;
    default:
      out_dtype = dtype;
      break;
  }
  output_ = std::make_shared<Tensor>(input_->n(), input_->c(), input_->h(),
                                     input_->w(), out_dtype);
}

void SCOPE::ActivationForward(Layer &layer) {
  const auto itype = input_->dtype();
  const auto otype = output_->dtype();
  auto activation = [=](auto x) -> decltype(x) {
    x = x > 0 ? x : 0;
    x = x > 6 ? 6 : x;
    return x;
  };

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
