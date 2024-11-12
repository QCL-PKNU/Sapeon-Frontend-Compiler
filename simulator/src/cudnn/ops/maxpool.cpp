#include "cudnn/ops/maxpool.hpp"

#define BASE CudnnOperation
#define NAME Maxpool
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
#include <numeric>
#include <string>
using std::string;
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
  handle_ = handle;

  InitOutputTensor();
  CreateDescriptors();
  SetDescriptors();
  AllocateMemory();
  OperationForward();
  GetOutput();
  DeAllocateMemory();
  DestroyDescriptors();

  return output_;
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::InitOutputTensor() {
  size_t sh = sampling_->stride_height();
  size_t sw = sampling_->stride_width();
  size_t wh = sampling_->window_height();
  size_t ww = sampling_->window_width();
  size_t pht = sampling_->padding_height_top();
  size_t phb = sampling_->padding_height_bottom();
  size_t pwl = sampling_->padding_width_left();
  size_t pwr = sampling_->padding_width_right();

  float height = ((input_->h() + (pht + phb) - wh) / sh) + 1;
  float width = ((input_->w() + (pwl + pwr) - ww) / sw) + 1;

  output_ = std::make_shared<Tensor>(
      input_->n(), input_->c(), static_cast<int>(height),
      static_cast<int>(width), dty::GetDataType<Type>());
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::CreateDescriptors() {
  cudnnCreateTensorDescriptor(&src_descriptor_);
  cudnnCreateTensorDescriptor(&dst_descriptor_);
  cudnnCreatePoolingDescriptor(&pl_descriptor_);
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::SetDescriptors() {
  cudnnSetTensor4dDescriptor(src_descriptor_, CUDNN_TENSOR_NCHW, DataType,
                             input_->n(), input_->c(),
                             input_->h() + sampling_->padding_height_top() +
                                 sampling_->padding_height_bottom(),
                             input_->w() + sampling_->padding_width_left() +
                                 sampling_->padding_width_right());
  cudnnSetTensor4dDescriptor(dst_descriptor_, CUDNN_TENSOR_NCHW, DataType,
                             output_->n(), output_->c(), output_->h(),
                             output_->w());
  cudnnSetPooling2dDescriptor(
      pl_descriptor_, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
      sampling_->window_height(), sampling_->window_width(), 0, 0,
      sampling_->stride_height(), sampling_->stride_width());
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::AllocateMemory() {
  const size_t input_height = input_->h() + sampling_->padding_height_top() +
                              sampling_->padding_height_bottom();
  const size_t input_width = input_->w() + sampling_->padding_width_left() +
                             sampling_->padding_width_right();
  const size_t input_dimension =
      input_->n() * input_->c() * input_height * input_width;
  const size_t input_size = input_dimension * sizeof(Type);

  Type data_background[input_dimension];
  std::fill(data_background, data_background + input_dimension,
            std::numeric_limits<Type>::lowest());

  cudaMalloc(&data_input_, input_size);
  cudaMemcpy(data_input_, data_background, input_size, cudaMemcpyHostToDevice);

  Type *data_input_dst =
      (Type *)data_input_ + (sampling_->padding_width_left());
  Type *data_input_src = (Type *)input_->data();

  for (int batch = 0; batch < input_->n(); batch++) {
    for (int i = 0; i < input_->c(); i++) {
      data_input_dst += input_width * sampling_->padding_height_top();
      for (int j = sampling_->padding_height_top();
           j < sampling_->padding_height_top() + input_->h(); j++) {
        // memcpy(data_input_dst, data_input_src, input_->w() * sizeof(Type) );
        cudaMemcpy(data_input_dst, data_input_src, input_->w() * sizeof(Type),
                   cudaMemcpyHostToDevice);
        data_input_dst += input_width;
        data_input_src += input_->w();
      }
      data_input_dst += input_width * sampling_->padding_height_bottom();
    }
  }
  /*
  cudaMalloc(&data_input_, input_->size());
  cudaMemcpy(data_input_, input_->data(), input_->size(),
             cudaMemcpyHostToDevice);
  */

  cudaMalloc(&data_output_, output_->size());
  cudaMemset(data_output_, 0, output_->size());
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::OperationForward() {
  float a = 1;
  float b = 1;

  cudnnPoolingForward(handle_, pl_descriptor_, &a, src_descriptor_, data_input_,
                      &b, dst_descriptor_, data_output_);
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::GetOutput() {
  cudaMemcpy(output_->data(), data_output_, output_->size(),
             cudaMemcpyDeviceToHost);
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::DeAllocateMemory() {
  cudaFree(data_output_);
  cudaFree(data_input_);
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::DestroyDescriptors() {
  cudnnDestroyPoolingDescriptor(pl_descriptor_);
  cudnnDestroyTensorDescriptor(dst_descriptor_);
  cudnnDestroyTensorDescriptor(src_descriptor_);
}
