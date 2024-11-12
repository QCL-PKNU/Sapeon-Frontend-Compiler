#include "cudnn/ops/sub.hpp"

#include "cudnn/common/binary_elementwise.hpp"

#define BASE CudnnOperation
#define NAME Sub
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
#include <vector>
using std::vector;
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
  if (layer.intermediate_activation() == nullptr) {
    inputs_ = layer.inputs();
  } else {
    inputs_ = vector<shared_ptr<Tensor>>();
    inputs_.push_back(layer.intermediate_activation());
  }
  handle_ = handle;

  InitOutputTensor();
  AllocateMemory();
  OperationForward();
  GetOutput();
  DeAllocateMemory();

  return output_;
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::InitOutputTensor() {
  std::vector<int64_t> out_vector;

  ComputeOutputShape("Sub", inputs_[0]->dimension().dims(),
                     inputs_[1]->dimension().dims(), out_vector);

  output_ = std::make_shared<Tensor>(out_vector, dty::GetDataType<Type>());
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::AllocateMemory() {
  cudaMalloc(&(data_input_[0]), inputs_[0]->size());
  cudaMemcpy(data_input_[0], inputs_[0]->data(), inputs_[0]->size(),
             cudaMemcpyHostToDevice);

  cudaMalloc(&(data_input_[1]), inputs_[1]->size());
  cudaMemcpy(data_input_[1], inputs_[1]->data(), inputs_[1]->size(),
             cudaMemcpyHostToDevice);

  cudaMalloc(&data_output_, output_->size());
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::OperationForward() {
  std::vector<int64_t> in_vector1 = inputs_[0]->dimension().dims();
  std::vector<int64_t> in_vector2 = inputs_[1]->dimension().dims();
  std::vector<int64_t> out_vector = output_->dimension().dims();
  cudaStream_t stream;

  cudnnGetStream(handle_, &stream);

  size_t count = output_->dimension().size();

  int32_t output_rank_or_simple_broadcast;
  fast_divmod fdm_H, fdm_C;
  TArray<int64_t> lhs_padded_strides, rhs_padded_strides;
  TArray<fast_divmod> fdm_output_strides;

  BinaryElementwiseBroadcastPrepare(
      in_vector1, in_vector2, out_vector, output_rank_or_simple_broadcast,
      fdm_H, fdm_C, lhs_padded_strides, rhs_padded_strides, fdm_output_strides);

  Impl_Sub<Type>(stream, output_rank_or_simple_broadcast, &lhs_padded_strides,
                 data_input_[0], &rhs_padded_strides, data_input_[1],
                 &fdm_output_strides, fdm_H, fdm_C, data_output_, count);
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::GetOutput() {
  cudaMemcpy(output_->data(), data_output_, output_->size(),
             cudaMemcpyDeviceToHost);
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::DeAllocateMemory() {
  cudaFree(data_output_);
  cudaFree(data_input_[0]);
  cudaFree(data_input_[1]);
}
