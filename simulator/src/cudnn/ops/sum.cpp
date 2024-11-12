#include "cudnn/ops/sum.hpp"

#include "cudnn/common/binary_elementwise.hpp"
#include "cudnn/common/cuda_common.hpp"
#include "cudnn/common/variadic_elementwise.hpp"
#include "cudnn/ops/sum_base.hpp"

#define BASE CudnnOperation
#define NAME Sum
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
#include <malloc.h>

#include "datatype.hpp"
#include "factory.hpp"
#include "glog/logging.h"
#include "network/descriptor.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cudnn_operation.hpp"
#include "utility.hpp"

namespace Cudnn {
/*
static bool kRegistered = Factory<BASE<DB>>::RegisterCreateFunction(
                              GET_STR(NAME), CLASS<FP64>::Create) &&
                          Factory<BASE<FL>>::RegisterCreateFunction(
                              GET_STR(NAME), CLASS<FP32>::Create) &&
                          Factory<BASE<SC>>::RegisterCreateFunction(
                              GET_STR(NAME), CLASS<INT8>::Create) &&
                          Factory<BASE<UC>>::RegisterCreateFunction(
                              GET_STR(NAME), CLASS<UINT8>::Create);
*/

static bool kRegistered = Factory<BASE<DB>>::RegisterCreateFunction(
                              GET_STR(NAME), CLASS<FP64>::Create) &&
                          Factory<BASE<FL>>::RegisterCreateFunction(
                              GET_STR(NAME), CLASS<FP32>::Create);

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

  inputs_count_ = layer.predecessors().size();
  if (inputs_count_ == 0) {
    inputs_count_ = 1;
  }

  InitOutputTensor();

  AllocateMemory();

  OperationForward();

  GetOutput();

  DeAllocateMemory();

  return output_;
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::InitOutputTensor() {
  std::vector<int64_t> out_vector = inputs_[0]->dimension().dims();

  if (inputs_count_ > 1) {
    std::vector<int64_t> in_vector1 = inputs_[0]->dimension().dims();

    for (size_t index = 1; index < inputs_count_; index++) {
      std::vector<int64_t> in_vector2 = inputs_[index]->dimension().dims();
      ComputeOutputShape("Sum", in_vector1, in_vector2, out_vector);
      in_vector1 = out_vector;
    }
  }
  output_ = std::make_shared<Tensor>(out_vector, dty::GetDataType<Type>());
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::AllocateMemory() {
  if (inputs_count_ > 1) {
    data_input_ = new Type *[inputs_count_];
    for (size_t index = 0; index < inputs_count_; index++) {
      cudaMalloc(&(data_input_[index]), inputs_[index]->size());
      cudaMemcpy(data_input_[index], inputs_[index]->data(),
                 inputs_[index]->size(), cudaMemcpyHostToDevice);
    }
    cudaMalloc(&data_output_, output_->size());
  }
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::OperationForward() {
  if (inputs_count_ == 1) {
    return;
  }
  size_t index = 0;
  std::vector<std::vector<int64_t>> in_vectors;
  for (index = 0; index < inputs_count_; index++) {
    in_vectors.push_back(inputs_[index]->dimension().dims());
  }

  std::vector<int64_t> out_vector = output_->dimension().dims();
  cudaStream_t stream;

  cudnnGetStream(handle_, &stream);

  for (index = 1; index < inputs_count_; index++) {
    if (in_vectors[index] != in_vectors[0]) {
      break;
    }
  }

  if (index == inputs_count_) {  // broadcast 가 필요 없음
    if (inputs_count_ == 2) {
      size_t count = output_->dimension().size();

      int32_t output_rank_or_simple_broadcast;
      fast_divmod fdm_H, fdm_C;
      TArray<int64_t> lhs_padded_strides, rhs_padded_strides;
      TArray<fast_divmod> fdm_output_strides;

      BinaryElementwiseBroadcastPrepare(
          in_vectors[0], in_vectors[1], out_vector,
          output_rank_or_simple_broadcast, fdm_H, fdm_C, lhs_padded_strides,
          rhs_padded_strides, fdm_output_strides);

      Impl_Add<Type>(stream, output_rank_or_simple_broadcast,
                     &lhs_padded_strides, data_input_[0], &rhs_padded_strides,
                     data_input_[1], &fdm_output_strides, fdm_H, fdm_C,
                     data_output_, count);

      return;
    }
    NoBroadcastBatchImplDispatchTarget<Type, DataType>(
        stream, inputs_count_, data_input_, data_output_, inputs_, output_);
  }

  if (inputs_count_ == 2) {
    BinaryImplDispatchTarget<Type, DataType>(stream, inputs_count_, data_input_,
                                             data_output_, inputs_, output_);
    return;
  }

  GeneralImplDispatchTarget<Type, DataType>(stream, inputs_count_, data_input_,
                                            data_output_, inputs_, output_);
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::GetOutput() {
  if (inputs_count_ == 1) {
    memcpy(output_->data(), inputs_[0]->data(), inputs_[0]->size());
  } else {
    cudaMemcpy(output_->data(), data_output_, output_->size(),
               cudaMemcpyDeviceToHost);
  }
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::DeAllocateMemory() {
  if (inputs_count_ > 1) {
    cudaFree(data_output_);
    for (size_t index = 0; index < inputs_count_; index++) {
      cudaFree(data_input_[index]);
    }
  }
}

}  // namespace Cudnn
