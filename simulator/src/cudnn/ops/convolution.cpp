#include "cudnn/ops/convolution.hpp"

#define BASE CudnnOperation
#define NAME Convolution
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
                              GET_STR(NAME), CLASS<INT8>::Create) &&
                          Factory<BASE<UC>>::RegisterCreateFunction(
                              GET_STR(NAME), CLASS<UINT8>::Create);

template <typename Type, cudnnDataType_t DataType>
unique_ptr<BASE<Type>> SCOPE::Create() {
  return make_unique<CLASS<Type, DataType>>();
}

template <typename Type, cudnnDataType_t DataType>
std::shared_ptr<Tensor> SCOPE::Forward(cudnnHandle_t &handle, Layer &layer) {
  input_ = layer.intermediate_activation() == nullptr
               ? layer.inputs(0)
               : layer.intermediate_activation();
  filter_ = layer.filter();
  convolution_ = layer.convolution();
  handle_ = handle;

  InitOutputTensor();
  CreateDescriptors();
  SetDescriptors();
  SetWorkspaceSize();
  AllocateMemory();
  OperationForward();
  GetOutput();
  DeAllocateMemory();
  DestroyDescriptors();

  if (layer.bias() == nullptr) {
    return output_;
  } else {
    layer.intermediate_activation(output_);
    auto p_operation = Factory<CudnnOperation<Type>>::CreateInstance("BiasAdd");
    return p_operation.get()->Forward(handle, layer);
  }
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::InitOutputTensor() {
  const size_t fh = filter_->h();
  const size_t fw = filter_->w();

  const size_t sh = convolution_->stride_height();
  const size_t sw = convolution_->stride_width();
  const size_t dh = convolution_->dilation_height();
  const size_t dw = convolution_->dilation_width();
  const size_t pht = convolution_->padding_height_top();
  const size_t phb = convolution_->padding_height_bottom();
  const size_t pwl = convolution_->padding_width_left();
  const size_t pwr = convolution_->padding_width_right();

  float height =
      ((input_->h() + (pht + phb) - fh - (fh - 1) * (dh - 1)) / sh) + 1;
  float width =
      ((input_->w() + (pwl + pwr) - fw - (fw - 1) * (dw - 1)) / sw) + 1;

  output_ = std::make_shared<Tensor>(
      input_->n(), filter_->n(), static_cast<int>(height),
      static_cast<int>(width), dty::GetDataType<Type>());
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::CreateDescriptors() {
  cudnnCreateTensorDescriptor(&src_descriptor_);
  cudnnCreateTensorDescriptor(&dst_descriptor_);
  cudnnCreateFilterDescriptor(&ft_descriptor_);
  cudnnCreateConvolutionDescriptor(&cv_descriptor_);

  if (convolution_->groups() > 1) {
    cudnnSetConvolutionGroupCount(cv_descriptor_, convolution_->groups());
  }
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::SetDescriptors() {
  cudnnSetTensor4dDescriptor(src_descriptor_, CUDNN_TENSOR_NCHW, DataType,
                             input_->n(), input_->c(),
                             input_->h() + convolution_->padding_height_top() +
                                 convolution_->padding_height_bottom(),
                             input_->w() + convolution_->padding_width_left() +
                                 convolution_->padding_width_right());
  cudnnSetTensor4dDescriptor(dst_descriptor_, CUDNN_TENSOR_NCHW, DataType,
                             output_->n(), output_->c(), output_->h(),
                             output_->w());
  cudnnSetFilter4dDescriptor(ft_descriptor_, DataType, CUDNN_TENSOR_NCHW,
                             filter_->n(), filter_->c(), filter_->h(),
                             filter_->w());
  cudnnSetConvolution2dDescriptor(
      cv_descriptor_, 0, 0, convolution_->stride_height(),
      convolution_->stride_width(), convolution_->dilation_height(),
      convolution_->dilation_width(), CUDNN_CROSS_CORRELATION, DataType);
  fwd_algo_mode_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
}

template <>
void CLASS<uint8_t, CUDNN_DATA_UINT8>::SetDescriptors() {
  cudnnSetTensor4dDescriptor(src_descriptor_, CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_UINT8, input_->n(), input_->c(),
                             input_->h() + convolution_->padding_height_top() +
                                 convolution_->padding_height_bottom(),
                             input_->w() + convolution_->padding_width_left() +
                                 convolution_->padding_width_right());
  cudnnSetTensor4dDescriptor(dst_descriptor_, CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_INT8, output_->n(), output_->c(),
                             output_->h(), output_->w());
  cudnnSetFilter4dDescriptor(ft_descriptor_, CUDNN_DATA_INT8, CUDNN_TENSOR_NCHW,
                             filter_->n(), filter_->c(), filter_->h(),
                             filter_->w());
  cudnnSetConvolution2dDescriptor(
      cv_descriptor_, 0, 0, convolution_->stride_height(),
      convolution_->stride_width(), convolution_->dilation_height(),
      convolution_->dilation_width(), CUDNN_CROSS_CORRELATION,
      CUDNN_DATA_INT32);
  fwd_algo_mode_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
}

template <>
void CLASS<int8_t, CUDNN_DATA_INT8>::SetDescriptors() {
  cudnnSetTensor4dDescriptor(src_descriptor_, CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_INT8, input_->n(), input_->c(),
                             input_->h() + convolution_->padding_height_top() +
                                 convolution_->padding_height_bottom(),
                             input_->w() + convolution_->padding_width_left() +
                                 convolution_->padding_width_right());
  cudnnSetTensor4dDescriptor(dst_descriptor_, CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_INT8, output_->n(), output_->c(),
                             output_->h(), output_->w());
  cudnnSetFilter4dDescriptor(ft_descriptor_, CUDNN_DATA_INT8, CUDNN_TENSOR_NCHW,
                             filter_->n(), filter_->c(), filter_->h(),
                             filter_->w());
  cudnnSetConvolution2dDescriptor(
      cv_descriptor_, 0, 0, convolution_->stride_height(),
      convolution_->stride_width(), convolution_->dilation_height(),
      convolution_->dilation_width(), CUDNN_CROSS_CORRELATION,
      CUDNN_DATA_INT32);
  fwd_algo_mode_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::SetWorkspaceSize() {
  cudnnGetConvolutionForwardWorkspaceSize(
      handle_, src_descriptor_, ft_descriptor_, cv_descriptor_, dst_descriptor_,
      fwd_algo_mode_, &workspace_size_);
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::AllocateMemory() {
  /*
  cudaMalloc(&data_input_, input_->size());
  cudaMemcpy(data_input_, input_->data(), input_->size(),
             cudaMemcpyHostToDevice);
  */

  int group_mx = convolution_->groups();

  if (group_mx <= 1) {
    group_mx = 1;
  }

  int input_height = input_->h() + convolution_->padding_height_top() +
                     convolution_->padding_height_bottom();
  int input_width = input_->w() + convolution_->padding_width_left() +
                    convolution_->padding_width_right();
  int input_size =
      input_->n() * input_->c() * input_height * input_width * sizeof(Type);
  cudaMalloc(&data_input_, input_size);
  cudaMemset(data_input_, 0, input_size);

  Type *data_input_dst =
      (Type *)data_input_ + (convolution_->padding_width_left());
  Type *data_input_src = (Type *)input_->data();

  for (int batch = 0; batch < input_->n(); batch++) {
    for (int i = 0; i < input_->c(); i++) {
      data_input_dst += input_width * convolution_->padding_height_top();
      for (int j = convolution_->padding_height_top();
           j < convolution_->padding_height_top() + input_->h(); j++) {
        // memcpy(data_input_dst, data_input_src, input_->w() * sizeof(Type) );
        cudaMemcpy(data_input_dst, data_input_src, input_->w() * sizeof(Type),
                   cudaMemcpyHostToDevice);
        data_input_dst += input_width;
        data_input_src += input_->w();
      }
      data_input_dst += input_width * convolution_->padding_height_bottom();
    }
  }

  cudaMalloc(&data_filter_, filter_->size());
  cudaMemcpy(data_filter_, filter_->data(), filter_->size(),
             cudaMemcpyHostToDevice);

  cudaMalloc(&data_workspace_, workspace_size_ * sizeof(Type));
  cudaMemset(data_workspace_, 0, workspace_size_ * sizeof(Type));

  cudaMalloc(&data_output_, output_->size());
  cudaMemset(data_output_, 0, output_->size());
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::OperationForward() {
  float a = 1;
  float b = 1;

  cudnnConvolutionForward(handle_, &a, src_descriptor_, data_input_,
                          ft_descriptor_, data_filter_, cv_descriptor_,
                          fwd_algo_mode_, data_workspace_, workspace_size_, &b,
                          dst_descriptor_, data_output_);
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::GetOutput() {
  cudaMemcpy(output_->data(), data_output_, output_->size(),
             cudaMemcpyDeviceToHost);
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::DeAllocateMemory() {
  cudaFree(data_workspace_);
  cudaFree(data_filter_);
  cudaFree(data_output_);
  cudaFree(data_input_);
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::DestroyDescriptors() {
  cudnnDestroyConvolutionDescriptor(cv_descriptor_);
  cudnnDestroyFilterDescriptor(ft_descriptor_);
  cudnnDestroyTensorDescriptor(dst_descriptor_);
  cudnnDestroyTensorDescriptor(src_descriptor_);
}
