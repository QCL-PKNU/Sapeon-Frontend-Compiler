#include "cpu/ops/reorg.hpp"

#define BASE CpuOperation
#define NAME Reorg
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

void SCOPE::Forward(Layer &layer, InferenceContext &ctx) {
  input_ = ctx.InputTensor(0);
  auto out_dtype = ctx.out_dtype();
  sampling_ = layer.sampling();

  InitOutputTensor(out_dtype);
  OperationForward();

  ctx.SetOutputTensor(output_);
}

bool SCOPE::CheckValidOperation(Layer &layer, Dimension input_dimension) {
  if (!layer.HasSamplingDescriptor()) {
    LOG(ERROR) << "Sampling descriptor not found";
    return false;
  }
  if (input_dimension.h() % layer.sampling()->stride_height() != 0) {
    LOG(ERROR) << "Reorg: The height of input tensor must be divisible "
                  "by stride_h, [N, H, W, C] = "
               << input_dimension.str()
               << ", stride_h = " << layer.sampling()->stride_height();
    return false;
  }
  if (input_dimension.w() % layer.sampling()->stride_width() != 0) {
    LOG(ERROR) << "Reorg: The width of input tensor must be divisible "
                  "by stride_w, [N, H, W, C] = "
               << input_dimension.str()
               << ", stride_w = " << layer.sampling()->stride_width();
    return false;
  }
  return true;
}

Dimension SCOPE::CalculateOutputDimension(Layer &layer,
                                          Dimension input_dimension) {
  int h_output =
      static_cast<int>(input_dimension.h() / layer.sampling()->stride_height());
  int w_output =
      static_cast<int>(input_dimension.w() / layer.sampling()->stride_width());
  int c_output = input_dimension.c() * layer.sampling()->stride_height() *
                 layer.sampling()->stride_width();

  return Dimension(input_dimension.n(), c_output, h_output, w_output);
}

void SCOPE::InitOutputTensor(dty::DataType dtype) {
  int h_output = input_->h() / sampling_->stride_height();
  int w_output = input_->w() / sampling_->stride_width();
  int c_output =
      input_->c() * sampling_->stride_height() * sampling_->stride_width();

  output_ = std::make_shared<Tensor>(input_->n(), c_output, h_output, w_output,
                                     dtype);
}

void SCOPE::OperationForward() {
  const dty::DataType itype = input_->dtype();
  const dty::DataType otype = output_->dtype();
  if (itype == dty::DataType::FP32 && otype == dty::DataType::FP32) {
    OperationForward<float>();
  } else if (itype == dty::DataType::FP64 && otype == dty::DataType::FP64) {
    OperationForward<double>();
  } else {
    DLOG(FATAL) << "reorg is not implemented for: " << dty::NameOf(itype);
  }
}

template <typename Type>
void SCOPE::OperationForward() {
  Type *input_data = input_->data<Type>();
  Type *output_data = output_->data<Type>();

  int input_n = input_->n();
  int input_c = input_->c();
  int input_h = input_->h();
  int input_w = input_->w();
  int output_c = output_->c();
  int output_h = output_->h();
  int output_w = output_->w();
  int stride_width = sampling_->stride_width();
  int stride_height = sampling_->stride_height();

#pragma omp parallel for schedule(static) default(shared) collapse(4)
  for (int b = 0; b < input_n; ++b) {
    for (int k = 0; k < output_c; ++k) {
      for (int j = 0; j < output_h; ++j) {
        for (int i = 0; i < output_w; ++i) {
          int new_channel = k % input_c;
          int offset = k / input_c;
          // TODO(oldman): check sampling_.stride_width() &
          // sampling_.stride_height()
          int new_width = i * stride_width + offset % stride_width;
          int new_height = j * stride_height + offset / stride_height;
          int in_index =
              new_width +
              input_w * (new_height + input_h * (new_channel + input_c * b));
          int out_index = i + output_w * (j + output_h * (k + output_c * b));

          output_data[out_index] = input_data[in_index];
        }
      }
    }
  }
}
