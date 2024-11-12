#include "cpu/ops/element_wise_multiplication.hpp"

#define BASE CpuOperation
#define NAME EWMul
#define CLASS cpu::NAME
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>

#include "datatype.hpp"
#include "factory.hpp"
#include "glog/logging.h"
#include "inference_context.hpp"
#include "network/descriptor.hpp"
#include "network/dimension.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cpu_operation.hpp"
#include "utility.hpp"
#include "x220/quant_config.hpp"

static bool kRegistered =
    Factory<BASE>::RegisterCreateFunction(GET_STR(NAME), CLASS::Create);

std::unique_ptr<BASE> SCOPE::Create() { return std::make_unique<CLASS>(); }

void SCOPE::Forward(Layer &layer, InferenceContext &ctx) {
  num_inputs_ = layer.predecessors().size();
  assert(num_inputs_ >= 1);
  inputs_.clear();
  for (int i = 0; i < num_inputs_; i++) {
    inputs_.push_back(ctx.InputTensor(i));
  }
  auto out_dtype = ctx.out_dtype();
  x220::QuantConfig &quant_config = layer.x220_quant_config();

  InitOutputTensor(out_dtype);
  OperationForward(quant_config);

  ctx.SetOutputTensor(output_);
}

bool SCOPE::CheckValidOperation(Layer &layer, Dimension input_dimension) {
  if (!layer.HasEwmulDescriptor()) {
    LOG(ERROR) << "EWMul Descriptor not found";
    return false;
  }
  return true;
}

Dimension SCOPE::CalculateOutputDimension(Layer &layer,
                                          Dimension input_dimension) {
  return input_dimension;
}

void SCOPE::InitOutputTensor(dty::DataType dtype) {
  output_ = std::make_shared<Tensor>(inputs_[0]->n(), inputs_[0]->c(),
                                     inputs_[0]->h(), inputs_[0]->w(), dtype);
}

void SCOPE::OperationForward(x220::QuantConfig &config) {
  const dty::DataType itype = inputs_.at(0)->dtype();
  const dty::DataType otype = output_->dtype();
  if (itype == dty::DataType::FP32 && otype == dty::DataType::FP32) {
    OperationForward<float>();
  } else if (itype == dty::DataType::FP64 && otype == dty::DataType::FP64) {
    OperationForward<double>();
  } else if (itype == dty::DataType::UINT8 && otype == dty::DataType::UINT8) {
    OperationQuantForward<uint8_t>(config);
  } else if (itype == dty::DataType::INT8 && otype == dty::DataType::INT8) {
    OperationQuantForward<int8_t>(config);
  } else if (itype == dty::DataType::INT16 && otype == dty::DataType::INT16) {
    OperationQuantForward<int16_t>(config);
  } else {
    DLOG(FATAL) << "element_wise_multiplication is not implemented for: "
                << dty::NameOf(itype);
  }
}

template <typename Type>
void SCOPE::OperationForward() {
  const auto *input_data_a = inputs_.at(0)->data<Type>();
  const auto *input_data_b = inputs_.at(1)->data<Type>();
  auto *output_data = output_->data<Type>();

  const double kMaxVal = static_cast<double>(std::numeric_limits<Type>::max());
  const double kMinVal =
      static_cast<double>(std::numeric_limits<Type>::lowest());
  const size_t kSize =
      output_->n() * output_->c() * output_->h() * output_->w();

#pragma omp parallel for simd
  for (int i = 0; i < kSize; ++i) {
    double immediate = static_cast<double>(input_data_a[i]) * input_data_b[i];
    immediate = std::min(std::max(immediate, kMinVal), kMaxVal);
    output_data[i] = static_cast<Type>(immediate);
  }
}

#ifndef CONFIDENTIAL_FEATURES
template <typename Type>
void SCOPE::OperationQuantForward(x220::QuantConfig &) {
  LOG(FATAL) << "Unsupported operation, please check your build configuration";
  exit(1);
}
#endif
