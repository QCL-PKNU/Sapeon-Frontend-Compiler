#include "cpu/ops/upsample.hpp"

#define BASE CpuOperation
#define NAME Upsample
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
#include "network/dimension.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cpu_operation.hpp"
#include "utility.hpp"

static bool kRegistered =
    Factory<BASE>::RegisterCreateFunction(GET_STR(NAME), CLASS::Create);

std::unique_ptr<BASE> SCOPE::Create() { return std::make_unique<CLASS>(); }

void SCOPE::Forward(Layer& layer, InferenceContext& ctx) {
  input_ = ctx.InputTensor(0);
  auto out_dtype = ctx.out_dtype();
  sampling_ = layer.sampling();

  InitOutputTensor(out_dtype);
  OperationForward();

  ctx.SetOutputTensor(output_);
}

bool SCOPE::CheckValidOperation(Layer& layer, Dimension input_dimension) {
  if (!layer.HasSamplingDescriptor()) {
    LOG(ERROR) << "Sampling descriptor not found";
    return false;
  }
  return true;
}

Dimension SCOPE::CalculateOutputDimension(Layer& layer,
                                          Dimension input_dimension) {
  return Dimension(input_dimension.n(), input_dimension.c(),
                   input_dimension.h() * layer.sampling()->stride_height(),
                   input_dimension.w() * layer.sampling()->stride_width());
}

void SCOPE::InitOutputTensor(dty::DataType dtype) {
  output_ = std::make_shared<Tensor>(
      input_->n(), input_->c(), input_->h() * sampling_->stride_height(),
      input_->w() * sampling_->stride_width(), dtype);
}

void SCOPE::OperationForward() {
  const dty::DataType itype = input_->dtype();
  const dty::DataType otype = output_->dtype();
  if (itype == dty::DataType::FP32 && otype == dty::DataType::FP32) {
    OperationForward<float>();
  } else if (itype == dty::DataType::FP64 && otype == dty::DataType::FP64) {
    OperationForward<double>();
  } else if (itype == dty::DataType::UINT8 && otype == dty::DataType::UINT8) {
    OperationForward<uint8_t>();
  } else if (itype == dty::DataType::INT8 && otype == dty::DataType::INT8) {
    OperationForward<int8_t>();
  } else if (itype == dty::DataType::INT16 && otype == dty::DataType::INT16) {
    OperationForward<int16_t>();
  } else {
    DLOG(FATAL) << "upsample is not implemented for" << dty::NameOf(itype);
  }
}

// clang-format off
// It is readable, but it is not optimized.
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

  const int stride_h = sampling_->stride_height();
  const int stride_w = sampling_->stride_width();

  std::fill(output_data, output_data + out_n * out_c * out_h * out_w, static_cast<Type>(0));

#pragma omp parallel for schedule(static) default(shared) collapse(4)
  for (int n = 0; n < out_n; ++n) {
    for (int c = 0; c < out_c; ++c) {
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          // Compute input index
          const int in_h_idx = h / stride_h;
          const int in_w_idx = w / stride_w;
          const int in_idx = (n * in_c * in_h * in_w) + (c * in_h * in_w) + (in_h_idx * in_w) + in_w_idx;

          // Compute output index and set value
          const int out_idx = (n * out_c * out_h * out_w) + (c * out_h * out_w) + (h * out_w) + w;
          output_data[out_idx] = input_data[in_idx];
        }
      }
    }
  }
}

/* It is possible to optimize this further by computing the offset once
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

  const int stride_h = sampling_->stride_height();
  const int stride_w = sampling_->stride_width();

  std::fill(output_data, output_data + out_n * out_c * out_h * out_w, static_cast<Type>(0));

  const int in_hw = in_h * in_w;
  const int out_hw = out_h * out_w;

  for (int n = 0; n < out_n; ++n) {
    for (int c = 0; c < out_c; ++c) {
      for (int h = 0, in_h_idx = 0; h < out_h; ++h, in_h_idx += stride_h) {
        // Optimize to compute offset once
        const int in_offset = ((n * in_c + c) * in_hw) + (in_h_idx * in_w);

        // Compute output index and set value
        for (int w = 0, in_w_idx = 0, out_idx = (n * out_c * out_hw) + (c * out_hw) + (h * out_w);
         w < out_w; ++w, in_w_idx += stride_w, ++out_idx) {
          const int in_idx = in_offset + in_w_idx;
          output_data[out_idx] = input_data[in_idx];
        }
      }
    }
  }
}
*/
// clang-format on
