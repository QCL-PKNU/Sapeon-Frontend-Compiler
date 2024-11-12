#include "cudnn/ops/clip.hpp"

#include "cudnn/ops/clip_impl.h"

#define BASE CudnnOperation
#define NAME Clip
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

namespace clip_internal {
template <typename T>
struct LowMax {
  constexpr static T low() { return std::numeric_limits<T>::lowest(); }
  constexpr static T max() { return std::numeric_limits<T>::max(); }
};
}  // namespace clip_internal

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
  output_ = std::make_shared<Tensor>(inputs_[0]->dimension().dims(),
                                     dty::GetDataType<Type>());
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::AllocateMemory() {
  cudaMalloc(&(data_input_[0]), inputs_[0]->size());
  cudaMemcpy(data_input_[0], inputs_[0]->data(), inputs_[0]->size(),
             cudaMemcpyHostToDevice);

  cudaMalloc(&data_output_, output_->size());
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::OperationForward() {
  cudaStream_t stream;

  cudnnGetStream(handle_, &stream);

  auto min_default = clip_internal::LowMax<Type>::low();
  auto max_default = clip_internal::LowMax<Type>::max();

  const Type *min_data = nullptr;
  const Type *max_data = nullptr;

  if (inputs_[1]) {
    if (inputs_[1]->dimension().size() == 1) {
      // min_data = inputs_[1]->data();
      min_default = inputs_[1]->data<Type>()[0];
    }
  }
  if (inputs_[2]) {
    if (inputs_[2]->dimension().size() == 1) {
      // max_data = inputs_[2]->data();
      max_default = inputs_[2]->data<Type>()[0];
    }
  }

  const size_t count = inputs_[0]->dimension().size();
  ClipImpl<Type>(stream, data_input_[0], data_output_, min_default, max_default,
                 count);
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
}
