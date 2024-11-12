#include "cpu/ops/lavgpool.hpp"

#define BASE CpuOperation
#define NAME Lavgpool
#define CLASS cpu::NAME
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

#include <cstdint>
#include <limits>
#include <memory>
#include <string>

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
  if (!layer.HasSamplingDescriptor()) {
    LOG(FATAL) << "Sampling descriptor not found";
  }
  sampling_ = layer.sampling();
  x220::QuantConfig &quant_config = layer.x220_quant_config();

  InitOutputTensor(out_dtype);
  OperationForward(quant_config);

  ctx.SetOutputTensor(output_);
}

bool SCOPE::CheckValidOperation(Layer &layer, Dimension input_dimension) {
  if (!layer.HasSamplingDescriptor()) {
    LOG(FATAL) << "Sampling descriptor not found";
    return false;
  }
  return true;
};

Dimension SCOPE::CalculateOutputDimension(Layer &layer,
                                          Dimension input_dimension) {
  const auto &sampling = layer.sampling();
  const size_t sh = sampling->stride_height();
  const size_t sw = sampling->stride_width();
  const size_t wh = sampling->window_height();
  const size_t ww = sampling->window_width();
  const size_t pht = sampling->padding_height_top();
  const size_t phb = sampling->padding_height_bottom();
  const size_t pwl = sampling->padding_width_left();
  const size_t pwr = sampling->padding_width_right();

  float height = ((input_dimension.h() + (pht + phb) - wh) / sh) + 1;
  float width = ((input_dimension.w() + (pwl + pwr) - ww) / sw) + 1;

  return {input_dimension.n(), input_dimension.c(), static_cast<size_t>(height),
          static_cast<size_t>(width)};
}

void SCOPE::InitOutputTensor(dty::DataType dtype) {
  const size_t sh = sampling_->stride_height();
  const size_t sw = sampling_->stride_width();
  const size_t wh = sampling_->window_height();
  const size_t ww = sampling_->window_width();
  const size_t pht = sampling_->padding_height_top();
  const size_t phb = sampling_->padding_height_bottom();
  const size_t pwl = sampling_->padding_width_left();
  const size_t pwr = sampling_->padding_width_right();

  float height = ((input_->h() + (pht + phb) - wh) / sh) + 1;
  float width = ((input_->w() + (pwl + pwr) - ww) / sw) + 1;

  output_ = std::make_shared<Tensor>(input_->n(), input_->c(),
                                     static_cast<int>(height),
                                     static_cast<int>(width), dtype);
}

void SCOPE::OperationForward(x220::QuantConfig &config) {
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

template <typename Type>
void SCOPE::OperationForward() {
  Type *input_data = input_->data<Type>();
  Type *output_data = output_->data<Type>();

  const size_t wl_offset = sampling_->padding_width_left();
  const size_t ht_offset = sampling_->padding_height_top();

  const size_t out_n = output_->n();
  const size_t out_c = output_->c();
  const size_t out_h = output_->h();
  const size_t out_w = output_->w();

  const size_t in_h = input_->h();
  const size_t in_w = input_->w();

  const size_t yy = sampling_->window_height();
  const size_t xx = sampling_->window_width();

  const size_t sh = sampling_->stride_height();
  const size_t sw = sampling_->stride_width();

#pragma omp parallel for schedule(static) default(shared) collapse(4)
  for (int n = 0; n < out_n; ++n) {
    for (int c = 0; c < out_c; ++c) {
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          size_t out_index = w + out_w * (h + out_h * (c + out_c * n));
          Type sum = 0;

          for (int y = 0; y < yy; ++y) {
            for (int x = 0; x < xx; ++x) {
              size_t cur_h = h * sh + y - ht_offset;
              size_t cur_w = w * sw + x - wl_offset;

              bool valid =
                  cur_h >= 0 && cur_h < in_h && cur_w >= 0 && cur_w < in_w;
              if (valid) {
                size_t idx = cur_w + in_w * (cur_h + in_h * (c + out_c * n));
                sum += input_data[idx];
              }
            }
          }
          // This value is divided by the number of elements in the window in
          // Quantization stage.
          output_data[out_index] = sum / (yy * xx);
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
void SCOPE::OperationQuantForward(x220::QuantConfig &) {
  LOG(FATAL) << "Unsupported operation, please check your build configuration";
  exit(1);
}
#endif
