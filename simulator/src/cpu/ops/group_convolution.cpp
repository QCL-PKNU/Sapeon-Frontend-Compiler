#include "cpu/ops/group_convolution.hpp"

#define BASE CpuOperation
#define NAME GroupConvolution
#define CLASS cpu::NAME
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>

#include "cpu/common/blas.hpp"
#include "cpu/common/gemm.hpp"
#include "cpu/common/im2col.hpp"
#include "datatype.hpp"
#include "factory.hpp"
#include "glog/logging.h"
#include "inference_context.hpp"
#include "network/descriptor.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cpu_operation.hpp"
#include "utility.hpp"
#include "x220/quant_config.hpp"

static bool kRegistered =
    Factory<BASE>::RegisterCreateFunction(GET_STR(NAME), CLASS::Create);

std::unique_ptr<BASE> SCOPE::Create() { return std::make_unique<CLASS>(); }

void SCOPE::Forward(Layer &layer, InferenceContext &ctx) {
  input_ = ctx.InputTensor(0);
  auto out_dtype = ctx.out_dtype();

  if (!layer.HasFilter()) {
    LOG(FATAL) << "Filter not found";
  }
  filter_ = layer.filter();
  if (!layer.HasBias()) {
    LOG(FATAL) << "Bias not found";
  }
  bias_ = layer.bias();
  if (!layer.HasConvolutionDescriptor()) {
    LOG(FATAL) << "Convolution descriptor not found";
  }
  convolution_ = layer.convolution();
  hidden_filter_ = filter_->dimension().dims()[0];
  x220::QuantConfig &quant_config = layer.x220_quant_config();

  assert(convolution_->dilation_height() == 1);

  InitOutputTensor(out_dtype);
  AllocateMemory(out_dtype);
  OperationForward(quant_config);

  ctx.SetOutputTensor(output_);
}

bool SCOPE::CheckValidOperation(Layer &layer, Dimension) {
  if (!layer.HasFilter()) {
    LOG(FATAL) << "Filter not found";
    return false;
  }
  if (!layer.HasBias()) {
    LOG(FATAL) << "Bias not found";
    return false;
  }
  if (!layer.HasConvolutionDescriptor()) {
    LOG(FATAL) << "Convolution descriptor not found";
    return false;
  }
  return true;
};

Dimension SCOPE::CalculateOutputDimension(Layer &layer,
                                          Dimension input_dimension) {
  const auto &filter = layer.filter();
  h_size_ = filter->h();
  w_size_ = filter->w();

  const size_t sh = layer.convolution()->stride_height();
  const size_t sw = layer.convolution()->stride_width();
  const size_t dh = layer.convolution()->dilation_height();
  const size_t dw = layer.convolution()->dilation_width();
  const size_t pht = layer.convolution()->padding_height_top();
  const size_t phb = layer.convolution()->padding_height_bottom();
  const size_t pwl = layer.convolution()->padding_width_left();
  const size_t pwr = layer.convolution()->padding_width_right();

  float height = ((input_dimension.h() + (pht + phb) - filter->h() -
                   (filter->h() - 1) * (dh - 1)) /
                  sh) +
                 1;
  float width = ((input_dimension.w() + (pwl + pwr) - filter->w() -
                  (filter->w() - 1) * (dw - 1)) /
                 sw) +
                1;
  return {input_dimension.n(), filter->n(), static_cast<size_t>(height),
          static_cast<size_t>(width)};
}

void SCOPE::InitOutputTensor(dty::DataType dtype) {
  h_size_ = filter_->h();
  w_size_ = filter_->w();

  const size_t sh = convolution_->stride_height();
  const size_t sw = convolution_->stride_width();
  const size_t dh = convolution_->dilation_height();
  const size_t dw = convolution_->dilation_width();
  const size_t pht = convolution_->padding_height_top();
  const size_t phb = convolution_->padding_height_bottom();
  const size_t pwl = convolution_->padding_width_left();
  const size_t pwr = convolution_->padding_width_right();

  float height =
      ((input_->h() + (pht + phb) - h_size_ - (h_size_ - 1) * (dh - 1)) / sh) +
      1;
  float width =
      ((input_->w() + (pwl + pwr) - w_size_ - (w_size_ - 1) * (dw - 1)) / sw) +
      1;

  output_ = std::make_shared<Tensor>(input_->n(), filter_->n(),
                                     static_cast<int64_t>(height),
                                     static_cast<int64_t>(width), dtype);
}

void SCOPE::AllocateMemory(dty::DataType dtype) {
  const size_t workspace_size =
      filter_->h() * filter_->w() * output_->h() * output_->w() * input_->c();

  data_workspace_ =
      std::make_shared<Tensor>(workspace_size * 2, dtype);  // need to handle bT
}

void SCOPE::OperationForward(x220::QuantConfig &config) {
  const dty::DataType itype = input_->dtype();
  if (itype == dty::DataType::FP32) {
    OperationForward<float>();
  } else if (itype == dty::DataType::FP64) {
    OperationForward<double>();
  } else if (itype == dty::DataType::UINT8) {
    OperationQuantForward<uint8_t, int8_t>(config);
  } else if (itype == dty::DataType::INT8) {
    OperationQuantForward<int8_t, int8_t>(config);
  } else if (itype == dty::DataType::INT16) {
    OperationQuantForward<int16_t, int16_t>(config);
  } else {
    LOG(ERROR) << "Not implemented forward data type" << '\n';
    exit(1);
  }
}

template <typename Type>
void SCOPE::OperationForward() {
  const size_t groups = convolution_->groups();

  const size_t nweights =
      h_size_ * w_size_ * hidden_filter_ * input_->c() / groups;
  const size_t m = hidden_filter_ / groups;                 // m
  const size_t filter_size = nweights / hidden_filter_;     // k
  const size_t spatial_size = output_->h() * output_->w();  // n

  const size_t sh = convolution_->stride_height();
  const size_t sw = convolution_->stride_width();
  const size_t pht = convolution_->padding_height_top();
  const size_t phb = convolution_->padding_height_bottom();
  const size_t pwl = convolution_->padding_width_left();
  const size_t pwr = convolution_->padding_width_right();
  const size_t dw = convolution_->dilation_width();
  const size_t dh = convolution_->dilation_height();

  memset(output_->data(), 0, output_->size());
  Type *work_data = data_workspace_->data<Type>();
  Type *input_data = input_->data<Type>();
  Type *output_data = output_->data<Type>();
  Type *filter_data = filter_->data<Type>();
  Type *bias_data = bias_->data<Type>();

  size_t i, j;

  for (i = 0; i < input_->n(); ++i) {
    for (j = 0; j < groups; ++j) {
      Type *filter_pos = filter_data + j * nweights / groups;
      Type *output_pos = output_data + (i * groups + j) * spatial_size * m;
      Type *im = input_data + (i * groups + j) * input_->h() * input_->w() *
                                  input_->c() / groups;

      im2col_cpu<Type>(im, input_->c() / groups, input_->h(), input_->w(),
                       h_size_, w_size_, pht, phb, pwl, pwr, sh, sw, dh, dw,
                       work_data);
      Gemm<Type>(0, 0, m, spatial_size, filter_size, 1, filter_pos, filter_size,
                 work_data, spatial_size, 1, output_pos, spatial_size);
      // Gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
    }
  }
  AddBias<Type>(output_data, bias_data, output_->n(), output_->c(),
                output_->h() * output_->w());
}

#ifndef CONFIDENTIAL_FEATURES
template <typename IType, typename WType>
void SCOPE::OperationQuantForward(x220::QuantConfig &) {
  LOG(FATAL) << "Unsupported operation, please check your build configuration";
  exit(1);
}
#endif
