#include "cudnn/ops/softmax.hpp"

#define BASE CudnnOperation
#define NAME Softmax
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
  output_ = std::make_shared<Tensor>(input_->n(), input_->c(), input_->h(),
                                     input_->w(), dty::GetDataType<Type>());
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::CreateDescriptors() {
  cudnnCreateTensorDescriptor(&src_descriptor_);
  cudnnCreateTensorDescriptor(&dst_descriptor_);
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::SetDescriptors() {
  cudnnSetTensor4dDescriptor(src_descriptor_, CUDNN_TENSOR_NCHW, DataType,
                             input_->n(), input_->c(), input_->h(),
                             input_->w());
  cudnnSetTensor4dDescriptor(dst_descriptor_, CUDNN_TENSOR_NCHW, DataType,
                             output_->n(), output_->c(), output_->h(),
                             output_->w());
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::AllocateMemory() {
  cudaMalloc(&data_input_, input_->size());
  cudaMemcpy(data_input_, input_->data(), input_->size(),
             cudaMemcpyHostToDevice);

  cudaMalloc(&data_output_, output_->size());
  cudaMemset(data_output_, 0, output_->size());
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::OperationForward() {
  float a = 1;
  float b = 1;

  cudnnSoftmaxForward(handle_, CUDNN_SOFTMAX_ACCURATE,
                      CUDNN_SOFTMAX_MODE_INSTANCE, &a, src_descriptor_,
                      data_input_, &b, dst_descriptor_, data_output_);
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
  cudnnDestroyTensorDescriptor(dst_descriptor_);
  cudnnDestroyTensorDescriptor(src_descriptor_);
}
