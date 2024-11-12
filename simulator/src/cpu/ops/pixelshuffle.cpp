#include "cpu/ops/pixelshuffle.hpp"

#define BASE CpuOperation
#define NAME Pixelshuffle
#define CLASS cpu::NAME
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

#include <cassert>
#include <memory>
#include <string>

#include "factory.hpp"
#include "glog/logging.h"
#include "inference_context.hpp"

using dty::DataType;
using x220::QuantConfig;

static bool kRegistered =
    Factory<BASE>::RegisterCreateFunction(GET_STR(NAME), CLASS::Create);

std::unique_ptr<BASE> SCOPE::Create() { return std::make_unique<CLASS>(); }

void SCOPE::Forward(Layer &layer, InferenceContext &ctx) {
  input_ = ctx.InputTensor(0);
  auto out_dtype = ctx.out_dtype();

  sampling_ = layer.sampling();
  QuantConfig &quant_config = layer.x220_quant_config();

  InitOutputTensor(out_dtype);
  OperationForward(quant_config);

  ctx.SetOutputTensor(output_);
}

bool SCOPE::CheckValidOperation(Layer &layer, Dimension input_dimension) {
  if (!layer.HasSamplingDescriptor()) {
    LOG(ERROR) << "Sampling descriptor not found";
    return false;
  }
  const size_t sw = layer.sampling()->stride_width();
  const size_t sh = layer.sampling()->stride_height();

  if (input_dimension.c() % (sh * sw) != 0) {
    LOG(ERROR) << "PixelShuffle: The channel of the input tensor must "
                  "be divisible by stride_h * stride_w, [N, H, W, C] = "
               << input_dimension.str()
               << ", stride_h * stride_w = " << sh * sw;
    return false;
  }
  return true;
}

Dimension SCOPE::CalculateOutputDimension(Layer &layer,
                                          Dimension input_dimension) {
  const size_t sw = layer.sampling()->stride_width();
  const size_t sh = layer.sampling()->stride_height();

  const int h_output = input_dimension.h() * sh;
  const int w_output = input_dimension.w() * sw;

  const int c_output = static_cast<int>(input_dimension.c() / (sh * sw));

  return Dimension(input_dimension.n(), c_output, h_output, w_output);
}

void SCOPE::InitOutputTensor(dty::DataType dtype) {
  // Get the stride width and height from the sampling descriptor
  const size_t sw = sampling_->stride_width();
  const size_t sh = sampling_->stride_height();

  // Calculate the output dimensions
  const int h_output = input_->h() * sh;
  const int w_output = input_->w() * sw;
  const int c_output = int(input_->c() / (sh * sw));

  // Create the output tensor with the calculated dimensions
  output_ = std::make_shared<Tensor>(input_->n(), c_output, h_output, w_output,
                                     dtype);

  // Check if the input dimensions are divisible by stride_h and stride_w
  assert(output_->w() == (input_->w() * sw));
  // stride_h is the same as stride_w
  assert(sw == sh);
}

void SCOPE::OperationForward(QuantConfig &config) {
  const dty::DataType itype = input_->dtype();
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
    LOG(ERROR) << "Not implemented forward data type" << '\n';
    exit(1);
  }
}

/**
 * Perform the pixel shuffle operation on the input tensor with the specified
 * upscale factor. The function rearranges elements from the input channels
 * to form the output channels, effectively performing a sub-pixel
 * convolution that increases the spatial resolution of the input tensor.
 */
template <typename Type>
void SCOPE::OperationForward() {
  Type *input_data = input_->data<Type>();
  Type *output_data = output_->data<Type>();
  int upscale_factor = sampling_->stride_width();

#pragma omp parallel for schedule(static) default(shared) collapse(5)
  for (int n = 0; n < input_->n(); ++n) {
    for (int oc = 0; oc < output_->c(); ++oc) {
      for (int h = 0; h < input_->h(); ++h) {
        for (int w = 0; w < input_->w(); ++w) {
          for (int kh = 0; kh < upscale_factor; ++kh) {
            for (int kw = 0; kw < upscale_factor; ++kw) {
              int ic = oc + (kh * upscale_factor + kw) * output_->c();
              int oh = h * upscale_factor + kh;
              int ow = w * upscale_factor + kw;

              output_data[((n * output_->c() + oc) * output_->h() + oh) *
                              output_->w() +
                          ow] =
                  input_data[((n * input_->c() + ic) * input_->h() + h) *
                                 input_->w() +
                             w];
            }
          }
        }
      }
    }
  }
}

template void SCOPE::OperationForward<int16_t>();
template void SCOPE::OperationForward<int8_t>();
template void SCOPE::OperationForward<uint8_t>();

#ifndef CONFIDENTIAL_FEATURES
template <typename Type>
void SCOPE::OperationQuantForward(QuantConfig &) {
  LOG(FATAL) << "Unsupported operation, please check your build configuration";
  exit(1);
}
#endif
