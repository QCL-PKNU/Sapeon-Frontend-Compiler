#include "gsl-lite.hpp"

#define NONE
#include "cudnn/common/binary_elementwise.hpp"
#include "cudnn/common/cuda_common.hpp"
#include "cudnn/common/reduction_ops.hpp"
#include "cudnn/common/tensor_shape.hpp"
#include "cudnn/common/variadic_elementwise.hpp"
#include "cudnn/ops/reduce_sum.hpp"

#define BASE CudnnOperation
#define NAME ReduceSum
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
#include "network/descriptor.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cudnn_operation.hpp"
#include "utility.hpp"

namespace Cudnn {
static bool kRegistered = Factory<BASE<DB>>::RegisterCreateFunction(
                              GET_STR(NAME), CLASS<FP64>::Create) &&
                          Factory<BASE<FL>>::RegisterCreateFunction(
                              GET_STR(NAME), CLASS<FP32>::Create) &&
                          Factory<BASE<FL>>::RegisterCreateFunction(
                              GET_STR(NAME), CLASS<FP16>::Create);

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

  if (layer.axes().size() > 0) {
    axes_ = ToShapeVector(layer.axes());
  } else {
    axes_.clear();
  }

  if (layer.keepdims() == std::numeric_limits<int>::lowest()) {
    keepdims_ = true;
  } else {
    keepdims_ = layer.keepdims();
  }

  if (layer.noop_with_empty_axes() == std::numeric_limits<int>::lowest()) {
    noop_with_empty_axes_ = false;
  } else {
    noop_with_empty_axes_ = layer.noop_with_empty_axes();
  }

  TensorShapeVector axes;

  TensorShape x_shape(inputs_[0]->dimension().dims());

  if (inputs_count_ == 2) {
    TensorShape axes_shape(inputs_[1]->dimension().dims());

    assert(axes_shape.NumDimensions() == 1);
    auto nDims = static_cast<size_t>(axes_shape[0]);
    const auto *data = (int64_t *)inputs_[1]->data();
    axes.assign(data, data + nDims);
  } else {
    axes.assign(axes_.begin(), axes_.end());
  }

  if (axes.empty() && noop_with_empty_axes_) {
    output_ = std::make_shared<Tensor>(inputs_[0]->dimension().dims(),
                                       dty::GetDataType<Type>());
    memcpy(output_->data(), inputs_[0]->data(), inputs_[0]->size());
    return output_;
  }

  PrepareReduceMetadata prepare_reduce_metadata;
  assert(PrepareForReduce(x_shape, keepdims_, axes, prepare_reduce_metadata));

  output_ = std::make_shared<Tensor>(
      prepare_reduce_metadata.squeezed_output_dims, dty::GetDataType<Type>());
  TensorShape output_shape(prepare_reduce_metadata.squeezed_output_dims);

  AllocateMemory();

  const bool fast_reduction = true;
  cudaStream_t stream;
  cudnnGetStream(handle_, &stream);

  ReduceComputeCore<Type, CUDNN_REDUCE_TENSOR_NO_INDICES>(
      stream, handle_, x_shape, data_input_[0], prepare_reduce_metadata,
      output_shape, data_output_, CUDNN_REDUCE_TENSOR_ADD, axes, false, false,
      false, 1, fast_reduction);

  GetOutput();

  DeAllocateMemory();

  return output_;
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::AllocateMemory() {
  data_input_ = new Type *[1];
  cudaMalloc(&(data_input_[0]), inputs_[0]->size());
  cudaMemcpy(data_input_[0], inputs_[0]->data(), inputs_[0]->size(),
             cudaMemcpyHostToDevice);
  cudaMalloc(&data_output_, output_->size());
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

}  // namespace Cudnn
