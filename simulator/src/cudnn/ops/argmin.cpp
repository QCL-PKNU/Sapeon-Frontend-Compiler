#include "gsl-lite.hpp"

#define NONE
#include <typeinfo>

#include "cudnn/common/binary_elementwise.hpp"
#include "cudnn/common/cuda_common.hpp"
#include "cudnn/common/cudnn_common.hpp"
#include "cudnn/common/reduction_ops.hpp"
#include "cudnn/common/tensor_shape.hpp"
#include "cudnn/common/variadic_elementwise.hpp"
#include "cudnn/ops/argmin.hpp"

#define BASE CudnnOperation
#define NAME ArgMin
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
                              GET_STR(NAME), CLASS<FP16>::Create) &&
                          Factory<BASE<SC>>::RegisterCreateFunction(
                              GET_STR(NAME), CLASS<INT8>::Create) &&
                          Factory<BASE<UC>>::RegisterCreateFunction(
                              GET_STR(NAME), CLASS<UINT8>::Create);

template <typename Type, cudnnDataType_t DataType>
unique_ptr<BASE<Type>> SCOPE::Create() {
  return make_unique<CLASS<Type, DataType>>();
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::SetOptions(Layer &layer) {
  axes_.clear();
  if (layer.axis() == std::numeric_limits<int>::lowest()) {
    axes_.push_back(0);
  } else {
    axes_.push_back(layer.axis());
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

  if (layer.select_last_index() == std::numeric_limits<int>::lowest()) {
    select_last_index_ = false;
  } else {
    select_last_index_ = layer.select_last_index();
  }
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

  SetOptions(layer);

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
                                       dty::GetDataType<int64_t>());

    memcpy(output_->data(), inputs_[0]->data(), inputs_[0]->size());
    return output_;
  }

  PrepareReduceMetadata prepare_reduce_metadata;
  assert(PrepareForReduce(x_shape, keepdims_, axes, prepare_reduce_metadata));

  output_ =
      std::make_shared<Tensor>(prepare_reduce_metadata.squeezed_output_dims,
                               dty::GetDataType<int64_t>());
  TensorShape output_shape(prepare_reduce_metadata.squeezed_output_dims);

  AllocateMemory();

  const bool fast_reduction = false;
  cudaStream_t stream;
  cudnnGetStream(handle_, &stream);

  ReduceComputeCore<Type, CUDNN_REDUCE_TENSOR_FLATTENED_INDICES>(
      stream, handle_, x_shape, data_input_[0], prepare_reduce_metadata,
      output_shape, (Type *)data_output_, CUDNN_REDUCE_TENSOR_MIN, axes, false,
      false, false, fast_reduction, 2);

  GetOutput();

  DeAllocateMemory();

  return output_;
}

template <typename InT, typename OutT>
void Impl_Cast(cudaStream_t stream, const InT *input_data, OutT *output_data,
               size_t count);

#define DEFINE_ARG_MAX_INT(TYPE, CUDNN_DATA_TYPE)                              \
  template <>                                                                  \
  std::shared_ptr<Tensor> Cudnn::ArgMin<TYPE, CUDNN_DATA_TYPE>::Forward(       \
      cudnnHandle_t &handle, Layer &layer) {                                   \
    inputs_ = layer.inputs();                                                  \
    handle_ = handle;                                                          \
                                                                               \
    inputs_count_ = layer.predecessors().size();                               \
    if (inputs_count_ == 0) {                                                  \
      inputs_count_ = 1;                                                       \
    }                                                                          \
                                                                               \
    SetOptions(layer);                                                         \
                                                                               \
    TensorShapeVector axes;                                                    \
                                                                               \
    TensorShape x_shape(inputs_[0]->dimension().dims());                       \
                                                                               \
    if (inputs_count_ == 2) {                                                  \
      TensorShape axes_shape(inputs_[1]->dimension().dims());                  \
                                                                               \
      assert(axes_shape.NumDimensions() == 1);                                 \
      auto nDims = static_cast<size_t>(axes_shape[0]);                         \
      const auto *data = (int64_t *)inputs_[1]->data();                        \
      axes.assign(data, data + nDims);                                         \
    } else {                                                                   \
      axes.assign(axes_.begin(), axes_.end());                                 \
    }                                                                          \
                                                                               \
    if (axes.empty() && noop_with_empty_axes_) {                               \
      output_ = std::make_shared<Tensor>(inputs_[0]->dimension().dims(),       \
                                         dty::GetDataType<int64_t>());         \
                                                                               \
      memcpy(output_->data(), inputs_[0]->data(), inputs_[0]->size());         \
      return output_;                                                          \
    }                                                                          \
                                                                               \
    PrepareReduceMetadata prepare_reduce_metadata;                             \
    assert(                                                                    \
        PrepareForReduce(x_shape, keepdims_, axes, prepare_reduce_metadata));  \
                                                                               \
    output_ =                                                                  \
        std::make_shared<Tensor>(prepare_reduce_metadata.squeezed_output_dims, \
                                 dty::GetDataType<int64_t>());                 \
    TensorShape output_shape(prepare_reduce_metadata.squeezed_output_dims);    \
                                                                               \
    AllocateMemory();                                                          \
                                                                               \
    const bool fast_reduction = false;                                         \
    cudaStream_t stream;                                                       \
    cudnnGetStream(handle_, &stream);                                          \
                                                                               \
    int64_t input_count = prepare_reduce_metadata.input_count;                 \
    int64_t output_count = prepare_reduce_metadata.output_count;               \
    auto &input_dims_cudnn = prepare_reduce_metadata.input_dims_cudnn;         \
    auto &output_dims_cudnn = prepare_reduce_metadata.output_dims_cudnn;       \
                                                                               \
    if (input_count == 0) {                                                    \
      assert(output_->dimension().dims().size() == 0);                         \
      GetOutput();                                                             \
                                                                               \
      DeAllocateMemory();                                                      \
                                                                               \
      return output_;                                                          \
    }                                                                          \
                                                                               \
    if (input_count == output_count) {                                         \
      memcpy(output_->data(), inputs_[0]->data(), inputs_[0]->size());         \
      DeAllocateMemory();                                                      \
      return output_;                                                          \
    }                                                                          \
                                                                               \
    assert(cudaMemsetAsync(data_output_, 0, output_->size(), stream) == 0);    \
                                                                               \
    size_t indices_bytes = 0;                                                  \
    size_t workspace_bytes = 0;                                                \
    CudnnTensor input_tensor;                                                  \
    CudnnTensor output_tensor;                                                 \
    CudnnReduceDescriptor reduce_desc;                                         \
                                                                               \
    typedef typename ToCudaType<TYPE>::MappedType CudaT;                       \
                                                                               \
    cudnnDataType_t cudnn_type_X = CUDNN_DATA_FLOAT;                           \
                                                                               \
    float *temp_X;                                                             \
    cudaMalloc(&temp_X, input_count * sizeof(float));                          \
                                                                               \
    Impl_Cast<CudaT, float>(stream,                                            \
                            reinterpret_cast<const CudaT *>(data_input_[0]),   \
                            temp_X, x_shape.Size());                           \
                                                                               \
    assert(reduce_desc.Set(CUDNN_REDUCE_TENSOR_MIN, cudnn_type_X,              \
                           CUDNN_REDUCE_TENSOR_NO_INDICES));                   \
    assert(input_tensor.Set(input_dims_cudnn, cudnn_type_X));                  \
    assert(output_tensor.Set(output_dims_cudnn, cudnn_type_X));                \
    assert(cudnnGetReductionIndicesSize(handle_, reduce_desc, input_tensor,    \
                                        output_tensor, &indices_bytes) == 0);  \
    assert(cudnnGetReductionWorkspaceSize(handle_, reduce_desc, input_tensor,  \
                                          output_tensor,                       \
                                          &workspace_bytes) == 0);             \
                                                                               \
    uint32_t *indices_cuda;                                                    \
    cudaMalloc(&indices_cuda, indices_bytes * sizeof(uint32_t));               \
                                                                               \
    CudaT *workspace_cuda;                                                     \
    cudaMalloc(&workspace_cuda, workspace_bytes * sizeof(CudaT));              \
                                                                               \
    const auto one = Consts<float>::One;                                       \
    const auto zero = Consts<float>::Zero;                                     \
                                                                               \
    float *temp_Y;                                                             \
    cudaMalloc(&temp_Y, output_count * sizeof(float));                         \
                                                                               \
    assert(cudnnReduceTensor(handle_, reduce_desc, indices_cuda,               \
                             indices_bytes, workspace_cuda, workspace_bytes,   \
                             &one, input_tensor, temp_X, &zero, output_tensor, \
                             temp_Y) == 0);                                    \
                                                                               \
    Impl_Cast<float, CudaT>(stream, temp_Y,                                    \
                            reinterpret_cast<CudaT *>(data_output_),           \
                            output_count);                                     \
                                                                               \
    cudaFree(temp_X);                                                          \
    cudaFree(indices_cuda);                                                    \
    cudaFree(workspace_cuda);                                                  \
                                                                               \
    GetOutput();                                                               \
                                                                               \
    DeAllocateMemory();                                                        \
                                                                               \
    return output_;                                                            \
  }

DEFINE_ARG_MAX_INT(UC, CUDNN_DATA_UINT8)
DEFINE_ARG_MAX_INT(SC, CUDNN_DATA_INT8)

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
