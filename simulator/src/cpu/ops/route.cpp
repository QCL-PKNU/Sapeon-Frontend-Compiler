#include "cpu/ops/route.hpp"

#define BASE CpuOperation
#define NAME Route
#define CLASS cpu::NAME
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

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

static bool kRegistered =
    Factory<BASE>::RegisterCreateFunction(GET_STR(NAME), CLASS::Create);

std::unique_ptr<BASE> SCOPE::Create() { return std::make_unique<CLASS>(); }

void SCOPE::Forward(Layer &layer, InferenceContext &ctx) {
  auto out_dtype = ctx.out_dtype();

  num_inputs_ = layer.predecessors().size();
  assert(num_inputs_ >= 1);
  inputs_.clear();
  for (int i = 0; i < num_inputs_; i++) {
    inputs_.push_back(ctx.InputTensor(i));
  }
  bool check = true;
  for (int i = 1; i < num_inputs_; ++i) {
    if (inputs_.at(i)->dtype() != inputs_.at(i - 1)->dtype()) {
      check = false;
    }
  }
  assert(check);

  InitOutputTensor(out_dtype);
  OperationForward();

  ctx.SetOutputTensor(output_);
}

bool SCOPE::CheckValidOperation(Layer &layer, Dimension input_dimension) {
  if (layer.input_dimensions().size() == 0) return false;
  int64_t n = layer.input_dimensions(0).n();
  int64_t h = layer.input_dimensions(0).h();
  int64_t w = layer.input_dimensions(0).w();
  for (auto dims : layer.input_dimensions()) {
    if (dims.n() != n || dims.h() != h || dims.w() != w) return false;
  }
  return true;
}

Dimension SCOPE::CalculateOutputDimension(Layer &layer,
                                          Dimension input_dimension) {
  int64_t out_c = 0;
  // This code only works if route is first sublayer
  // TODO: check this assumption is true or not
  for (auto dims : layer.input_dimensions()) {
    out_c += dims.c();
  }
  auto dims = layer.input_dimensions(0);
  return Dimension(dims.n(), out_c, dims.h(), dims.w());
}

void SCOPE::InitOutputTensor(dty::DataType dtype) {
  for (int i = 0; i < num_inputs_; i++) {
    num_channels_ += inputs_[i]->c();
  }
  output_ = std::make_shared<Tensor>(inputs_[0]->n(), num_channels_,
                                     inputs_[0]->h(), inputs_[0]->w(), dtype);
}

void SCOPE::OperationForward() {
  const dty::DataType itype = inputs_.at(0)->dtype();
  const dty::DataType otype = output_->dtype();
  if (itype == otype) {
    switch (itype) {
      case dty::DataType::FP32:
        OperationForward<float>();
        break;
      case dty::DataType::FP64:
        OperationForward<double>();
        break;
      case dty::DataType::INT16:
        OperationForward<int16_t>();
        break;
      case dty::DataType::INT8:
        OperationForward<int8_t>();
        break;
      case dty::DataType::UINT8:
        OperationForward<uint8_t>();
        break;
      default:
        DLOG(FATAL) << "route is not implemented for: " << dty::NameOf(itype);
    }
  } else {
    DLOG(FATAL) << "route is not implemented for: " << dty::NameOf(itype);
  }
}

template <typename Type>
void SCOPE::OperationForward() {
  Type *input_data;
  Type *output_data = output_->data<Type>();

  size_t offset = 0;
  for (int i = 0; i < num_inputs_; i++) {
    input_data = inputs_[i]->data<Type>();
    const int offset_h = inputs_[i]->w();
    const int offset_c = inputs_[i]->h() * offset_h;
    const int offset_n = inputs_[i]->c() * offset_c;
    const int out_offset_n = num_channels_ * offset_c;
    for (int n = 0; n < inputs_[i]->n(); ++n)
      for (int c = 0; c < inputs_[i]->c(); ++c)
        for (int h = 0; h < inputs_[i]->h(); ++h)
          for (int w = 0; w < inputs_[i]->w(); ++w) {
            size_t idx_in = n * offset_n + c * offset_c + h * offset_h + w;
            size_t idx_out =
                n * out_offset_n + (c + offset) * offset_c + h * offset_h + w;
            output_data[idx_out] = input_data[idx_in];
          }
    offset += inputs_[i]->c();
  }
}
