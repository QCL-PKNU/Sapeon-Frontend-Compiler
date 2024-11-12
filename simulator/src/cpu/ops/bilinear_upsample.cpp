#include "cpu/ops/bilinear_upsample.hpp"

#define BASE CpuOperation
#define NAME BilinearUpsample
#define CLASS cpu::NAME
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

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

using x220::QuantConfig;

static bool kRegistered =
    Factory<BASE>::RegisterCreateFunction(GET_STR(NAME), CLASS::Create);

std::unique_ptr<BASE> SCOPE::Create() { return std::make_unique<CLASS>(); }

void SCOPE::Forward(Layer& layer, InferenceContext& ctx) {
  input_ = ctx.InputTensor(0);
  auto out_dtype = ctx.out_dtype();
  if (!layer.HasSamplingDescriptor()) {
    LOG(FATAL) << "Sampling descriptor not found";
  }
  sampling_ = layer.sampling();
  QuantConfig& quant_config = layer.x220_quant_config();

  InitOutputTensor(out_dtype);
  OperationForward(quant_config);

  ctx.SetOutputTensor(output_);
}

bool SCOPE::CheckValidOperation(Layer& layer, Dimension) {
  if (!layer.HasSamplingDescriptor()) {
    LOG(FATAL) << "Sampling descriptor not found";
    return false;
  }
  return true;
};

Dimension SCOPE::CalculateOutputDimension(Layer& layer,
                                          Dimension input_dimension) {
  return {input_dimension.n(), input_dimension.c(),
          input_dimension.h() * layer.sampling()->stride_height(),
          input_dimension.w() * layer.sampling()->stride_width()};
}

void SCOPE::InitOutputTensor(dty::DataType dtype) {
  output_ = std::make_shared<Tensor>(
      input_->n(), input_->c(), input_->h() * sampling_->stride_height(),
      input_->w() * sampling_->stride_width(), dtype);
}

void SCOPE::OperationForward(QuantConfig& config) {
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
  Type* input_data = input_->data<Type>();
  Type* output_data = output_->data<Type>();

  const int out_n = output_->n();
  const int out_c = output_->c();
  const int out_h = output_->h();
  const int out_w = output_->w();

  const int in_c = input_->c();
  const int in_h = input_->h();
  const int in_w = input_->w();

  for (int n = 0; n < out_n; ++n) {
    int n_in_offset = n * in_c * in_h * in_w;
    int n_out_offset = n * out_c * out_h * out_w;

    for (int c = 0; c < out_c; ++c) {
      int c_in_offset = c * in_h * in_w;
      int c_out_offset = c * out_h * out_w;

      for (int h = 0; h < out_h; ++h) {
        // Calculate input position of the output pixel to point to the center
        float input_y = static_cast<float>((h + 0.5) * (in_h) / out_h - 0.5);
        int y0 = static_cast<int>(input_y);
        int y1 = std::min(y0 + 1, in_h - 1);
        float dy = input_y - y0;

        for (int w = 0; w < out_w; ++w) {
          float input_x = static_cast<float>((w + 0.5) * (in_w) / out_w - 0.5);
          int x0 = static_cast<int>(input_x);
          int x1 = std::min(x0 + 1, in_w - 1);
          float dx = input_x - x0;

          // Compute input indices
          int idx00 = n_in_offset + c_in_offset + (y0 * in_w) + x0;
          int idx01 = n_in_offset + c_in_offset + (y1 * in_w) + x0;
          int idx10 = n_in_offset + c_in_offset + (y0 * in_w) + x1;
          int idx11 = n_in_offset + c_in_offset + (y1 * in_w) + x1;

          // Bilinear interpolation
          Type val = (1 - dx) * (1 - dy) * input_data[idx00] +
                     (1 - dx) * dy * input_data[idx01] +
                     dx * (1 - dy) * input_data[idx10] +
                     dx * dy * input_data[idx11];

          // Compute output index and set value
          int out_idx = n_out_offset + c_out_offset + (h * out_w) + w;
          output_data[out_idx] = val;
        }
      }
    }
  }
}

#ifndef CONFIDENTIAL_FEATURES
template <typename Type>
void SCOPE::OperationQuantForward(QuantConfig&) {
  LOG(FATAL) << "Unsupported operation, please check your build configuration";
  exit(1);
}
#endif
