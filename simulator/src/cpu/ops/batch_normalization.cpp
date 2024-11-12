#include "cpu/ops/batch_normalization.hpp"

#define BASE CpuOperation
#define NAME BatchNormalization
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

  mean_ = layer.mean();
  scale_ = layer.scale();
  variance_ = layer.variance();
  epsilon_ = layer.epsilon();

  InitOutputTensor(out_dtype);
  OperationForward();

  ctx.SetOutputTensor(output_);
}

bool SCOPE::CheckValidOperation(Layer &layer, Dimension input_dimension) {
  if (!layer.HasMean()) {
    LOG(ERROR) << "Mean not found";
    return false;
  }
  if (!layer.HasScale()) {
    LOG(ERROR) << "Scale not found";
    return false;
  }
  if (!layer.HasVariance()) {
    LOG(ERROR) << "Variance not found";
    return false;
  }
  if (layer.mean()->dimension().size() != input_dimension.c()) {
    LOG(ERROR) << "Mismatch: mean size = " << layer.mean()->dimension().size()
               << ", input channels = " << input_dimension.c();
    return false;
  }
  if (layer.scale()->dimension().size() != input_dimension.c()) {
    LOG(ERROR) << "Mismatch: scale size = " << layer.scale()->dimension().size()
               << ", input channels = " << input_dimension.c();
    return false;
  }
  if (layer.variance()->dimension().size() != input_dimension.c()) {
    LOG(ERROR) << "Mismatch: variance size = "
               << layer.scale()->dimension().size()
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
    DLOG(FATAL) << "batch_normalization is not implemented for: "
                << dty::NameOf(itype);
  }
}

template <typename Type>
void SCOPE::OperationForward() {
  size_t b, f, i;
  size_t batch = input_->dimension().dims()[0];    // n
  size_t filters = input_->dimension().dims()[1];  // c
  size_t spatial =
      input_->dimension().dims()[2] * input_->dimension().dims()[3];  // size

  Type *x = input_->data<Type>();
  Type *output = output_->data<Type>();
  Type *mean = mean_->data<Type>();
  Type *variance = variance_->data<Type>();
  Type *scales = scale_->data<Type>();

  memcpy(output, x, input_->size());
  NormalizeCpu<Type>(output, mean, variance, batch, filters, spatial, epsilon_);

  ScaleBias<Type>(output, scales, batch, filters, spatial);
}
