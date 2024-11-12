#include "cpu/ops/element_wise_add.hpp"

#define BASE CpuOperation
#define NAME EWAdd
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
  if (!layer.HasEwaddDescriptor()) {
    LOG(ERROR) << "EWAdd descriptor not found";
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
    DLOG(FATAL) << "element_wise_add is not implemented for: "
                << dty::NameOf(itype);
  }
}

template <typename Type>
void SCOPE::OperationForward() {
  Type *input_data;
  Type *output_data = output_->data<Type>();
  for (int i = 0; i < output_->dimension().size(); i++) {
    output_data[i] = 0;
  }

  // EWAdd is an operation that sums tensors of the same dimension.
  // Therefore, it outputs a tensor with the same dimension as the input.
  const int offset_h = output_->w();
  const int offset_c = output_->h() * offset_h;
  const int offset_n = output_->c() * offset_c;
  for (int i = 0; i < num_inputs_; i++) {
    input_data = inputs_[i]->data<Type>();
    for (int n = 0; n < inputs_[i]->n(); ++n)
      for (int c = 0; c < inputs_[i]->c(); ++c)
        for (int h = 0; h < inputs_[i]->h(); ++h)
          for (int w = 0; w < inputs_[i]->w(); ++w) {
            size_t idx = n * offset_n + c * offset_c + h * offset_h + w;
            output_data[idx] += input_data[idx];
          }
  }
}

#ifndef CONFIDENTIAL_FEATURES
template <typename Type>
void SCOPE::OperationQuantForward(x220::QuantConfig &config) {
  const size_t size = output_->n() * output_->c() * output_->h() * output_->w();

  const int qmax = config.oqmax();
  const int qmin = config.oqmin();

  const int a_mul = config.shortcut().a_mul;
  const int b_mul = config.shortcut().b_mul;
  const int rsh = config.shortcut().rsh;
  // const int64_t bias  = x220_quant_config_.shortcut.bias;
  const int64_t bias = (rsh >= 1 ? (0x1 << (rsh - 1)) : 0);

  Type *input_data_a = inputs_.at(0)->data<Type>();
  Type *input_data_b = inputs_.at(1)->data<Type>();
  Type *output_data = output_->data<Type>();

#pragma omp parallel for simd
  for (int i = 0; i < size; ++i) {
    int a = input_data_a[i];
    int b = input_data_b[i];

    int o = (int64_t(a * a_mul) + int64_t(b * b_mul) + bias) >> rsh;
    o = std::min(qmax, std::max(qmin, o));
    output_data[i] = o;
  }
}
#endif
