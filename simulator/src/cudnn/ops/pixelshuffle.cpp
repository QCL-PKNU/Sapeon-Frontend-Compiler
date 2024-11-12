#include "cudnn/ops/pixelshuffle.hpp"

#define BASE CudnnOperation
#define NAME Pixelshuffle
#define CLASS Cudnn::NAME
#define SCOPE CLASS<Type, DataType>
#define DB double
#define FL float
#define UC uint8_t
#define SC int8_t
#define FP64 DB, CUDNN_DATA_DOUBLE
#define FP32 FL, CUDNN_DATA_FLOAT
#define FP16 FL, CUDNN_DATA_HALF
#define UINT8 UC, CUDNN_DATA_UINT8
#define INT8 SC, CUDNN_DATA_INT8
#define STR(x) #x
#define GET_STR(x) STR(x)

#include <memory>
using std::make_shared;
using std::make_unique;
using std::shared_ptr;
using std::unique_ptr;
#include <stdexcept>
using std::logic_error;
#include <string>
using std::string;
using std::to_string;
#include <cudnn.h>

#include "datatype.hpp"
#include "factory.hpp"
#include "network/descriptor.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cudnn_operation.hpp"
#include "utility.hpp"

static bool kRegistered = Factory<BASE<DB>>::RegisterCreateFunction(
                              GET_STR(NAME), CLASS<FP64>::Create) &&
                          Factory<BASE<FL>>::RegisterCreateFunction(
                              GET_STR(NAME), CLASS<FP32>::Create) &&
                          Factory<BASE<SC>>::RegisterCreateFunction(
                              GET_STR(NAME), CLASS<INT8>::Create);

template <typename Type, cudnnDataType_t DataType>
unique_ptr<BASE<Type>> SCOPE::Create() {
  return make_unique<CLASS<Type, DataType>>();
}

template <typename Type, cudnnDataType_t DataType>
std::shared_ptr<Tensor> SCOPE::Forward(cudnnHandle_t &handle, Layer &layer) {
  input_ = layer.intermediate_activation() == nullptr
               ? layer.inputs(0)
               : layer.intermediate_activation();
  sampling_ = layer.sampling();

  // jieun TODO
  // assert: The depth of the input tensor must be divisible by
  // stride_w*stride_h
  InitOutputTensor();
  OperationForward();

  return output_;
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::InitOutputTensor() {
  string input_dim = "";

  if (input_->c() % (sampling_->stride_height() * sampling_->stride_width()) !=
      0) {
    input_dim = "[" + to_string(input_->h()) + ", " + to_string(input_->w()) +
                ", " + to_string(input_->c()) + "]";
    throw logic_error(
        "PixelShuffle: The channel of the input tensor must "
        "be divisible by stride_h*stride_w, [H, W, C] = " +
        input_dim);
  }

  int h_output = input_->h() * sampling_->stride_height();
  int w_output = input_->w() * sampling_->stride_width();

  int c_output = int(input_->c() /
                     (sampling_->stride_height() * sampling_->stride_width()));

  output_ = std::make_shared<Tensor>(input_->n(), c_output, h_output, w_output,
                                     dty::GetDataType<Type>());
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::OperationForward() {
  Type *input_data = input_->data<Type>();
  Type *output_data = output_->data<Type>();

  Type **p_source = (Type **)calloc(
      sampling_->stride_height() * sampling_->stride_width(), sizeof(Type *));
  Type *p_destination;
  for (int n = 0; n < input_->n(); ++n) {
    for (int c = 0; c < output_->c(); ++c) {
      for (int idx_stride = 0;
           idx_stride < sampling_->stride_height() * sampling_->stride_width();
           ++idx_stride) {
        p_source[idx_stride] = input_data + input_->h() * input_->w() *
                                                (c + output_->c() * idx_stride);
      }
      p_destination = output_data + output_->h() * output_->w() * c;
      for (int h = 0; h < input_->h(); ++h) {
        for (int w = 0; w < input_->w(); ++w) {
          int idx = 0;
          for (int stride_height = 0;
               stride_height < sampling_->stride_height(); ++stride_height) {
            for (int stride_width = 0; stride_width < sampling_->stride_width();
                 ++stride_width) {
              *(p_destination +
                (output_->w() *
                 (sampling_->stride_height() * h + stride_height)) +
                sampling_->stride_width() * w + stride_width) =
                  *p_source[idx++]++;
            }
          }
        }
      }
    }
  }
  free(p_source);
}
