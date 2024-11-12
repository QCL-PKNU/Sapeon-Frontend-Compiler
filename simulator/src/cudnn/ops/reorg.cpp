#include "cudnn/ops/reorg.hpp"

#define BASE CudnnOperation
#define NAME Reorg
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

  InitOutputTensor();
  OperationForward();

  return output_;
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::InitOutputTensor() {
  string input_dim = "";

  if (input_->h() % sampling_->stride_height() != 0) {
    input_dim = "[" + to_string(input_->h()) + ", " + to_string(input_->w()) +
                ", " + to_string(input_->c()) + "]";
    throw logic_error(
        "Reorg: The height of input tensor must be divisible "
        "by stride_h, [H, W, C] = " +
        input_dim);
  } else if (input_->w() % sampling_->stride_width() != 0) {
    input_dim = "[" + to_string(input_->h()) + ", " + to_string(input_->w()) +
                ", " + to_string(input_->c()) + "]";
    throw logic_error(
        "Reorg: The width of input tensor must be divisible "
        "by stride_w, [H, W, C] = " +
        input_dim);
  }

  int h_output = int(input_->h() / sampling_->stride_height());
  int w_output = int(input_->w() / sampling_->stride_width());
  int c_output =
      input_->c() * sampling_->stride_height() * sampling_->stride_width();

  output_ = std::make_shared<Tensor>(input_->n(), c_output, h_output, w_output,
                                     dty::GetDataType<Type>());
}

template <typename Type, cudnnDataType_t DataType>
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
