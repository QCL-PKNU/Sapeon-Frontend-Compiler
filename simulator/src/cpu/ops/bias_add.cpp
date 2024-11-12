#include "cpu/ops/bias_add.hpp"

#define BASE CpuOperation
#define NAME BiasAdd
#define CLASS cpu::NAME
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

#include <cmath>
#include <memory>
#include <string>

#include "cpu/common/blas.hpp"
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
  input_ = ctx.InputTensor(0);
  auto out_dtype = ctx.out_dtype();
  bias_ = layer.bias();

  InitOutputTensor(out_dtype);
  OperationForward();

  ctx.SetOutputTensor(output_);
}

bool SCOPE::CheckValidOperation(Layer &layer, Dimension input_dimension) {
  if (!layer.HasBias()) {
    LOG(ERROR) << "Bias not found";
    return false;
  }
  if (layer.bias()->dimension().size() != input_dimension.c()) {
    LOG(ERROR) << "Mismatch: bias size = " << layer.bias()->dimension().size()
               << ", input channels = " << input_dimension.c();
    return false;
  }
  return true;
}

Dimension SCOPE::CalculateOutputDimension(Layer &layer,
                                          Dimension input_dimension) {
  return input_dimension;
}

void SCOPE::InitOutputTensor(dty::DataType dtype) {
  output_ = std::make_shared<Tensor>(input_->n(), input_->c(), input_->h(),
                                     input_->w(), dtype);
}

void SCOPE::OperationForward() {
  const dty::DataType itype = input_->dtype();
  const dty::DataType otype = output_->dtype();
  if (itype == dty::DataType::FP32 && otype == dty::DataType::FP32) {
    OperationForward<float>();
  } else if (itype == dty::DataType::FP64 && otype == dty::DataType::FP64) {
    OperationForward<double>();
  } else {
    DLOG(FATAL) << "bias_add is not implemented for: " << dty::NameOf(itype);
  }
}

template <typename Type>
void SCOPE::OperationForward() {
  size_t batch = input_->dimension().dims()[0];    // n
  size_t filters = input_->dimension().dims()[1];  // c
  size_t spatial =
      input_->dimension().dims()[2] * input_->dimension().dims()[3];  // size

  Type *x = input_->data<Type>();
  Type *output = output_->data<Type>();
  Type *bias = bias_->data<Type>();

  memcpy(output, x, output_->size());

  AddBias<Type>(output, bias, batch, filters, spatial);
}
