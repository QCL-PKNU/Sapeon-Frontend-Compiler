#include "cudnn/ops/layer_norm.hpp"

#include "cudnn/common/cuda_common.hpp"
#include "cudnn/common/tensor_shape.hpp"
#include "cudnn/ops/clip_impl.h"
#include "cudnn/ops/layer_norm_impl.cuh"
#include "gsl-lite.hpp"

#define BASE CudnnOperation
#define NAME LayerNorm
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
                              GET_STR(NAME), CLASS<FP32>::Create);

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

  inputs_count_ = layer.predecessors().size();
  if (inputs_count_ == 0) {
    inputs_count_ = 1;
  }

  if (layer.axis() == std::numeric_limits<int>::lowest()) {
    axis_ = -1;
  } else {
    axis_ = layer.axis();
  }

  if (layer.stash_type() == std::numeric_limits<int>::lowest()) {
    stash_type_ = 1;
  } else {
    stash_type_ = layer.stash_type();
  }

  if (layer.epsilon() != layer.epsilon()) {
    epsilon_ = 1e-05;
  } else {
    epsilon_ = layer.epsilon();
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
  output_ = std::make_shared<Tensor>(inputs_[0]->dimension().dims(),
                                     dty::GetDataType<Type>());
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::AllocateMemory() {
  for (size_t index = 0; index < inputs_count_; index++) {
    cudaMalloc(&(data_input_[index]), inputs_[index]->size());
    cudaMemcpy(data_input_[index], inputs_[index]->data(),
               inputs_[index]->size(), cudaMemcpyHostToDevice);
  }

  cudaMalloc(&data_output_, output_->size());
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::OperationForward() {
  cudaStream_t stream;

  cudnnGetStream(handle_, &stream);

  typedef typename ToCudaType<Type>::MappedType CudaT;
  typedef typename ToCudaType<Type>::MappedType CudaU;
  typedef typename ToCudaType<Type>::MappedType CudaV;

  auto X_data = reinterpret_cast<const CudaT *>(data_input_[0]);
  auto scale_data = reinterpret_cast<const CudaV *>(data_input_[1]);
  auto bias_data = (nullptr == data_input_[2])
                       ? nullptr
                       : reinterpret_cast<const CudaV *>(data_input_[2]);

  TensorShape x_shape(inputs_[0]->dimension().dims());
  const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());

  int n1 = gsl::narrow<int>(x_shape.SizeToDimension(axis));
  int n2 = gsl::narrow<int>(x_shape.SizeFromDimension(axis));

  const auto scale_size = inputs_[1]->dimension().size();
  const auto bias_size = (bias_data) ? inputs_[2]->dimension().size() : 0;

  assert(!(n2 == 1 || scale_size != n2 || (bias_data && bias_size != n2)));

  // Outputs
  auto Y_data = reinterpret_cast<CudaV *>(data_output_);

  // Mean and variance
  std::vector<int64_t> mean_inv_std_var_dim;
  for (int i = 0; i < static_cast<int>(x_shape.NumDimensions()); ++i) {
    if (i < axis) {
      mean_inv_std_var_dim.emplace_back(x_shape.GetDims()[i]);
    } else {
      mean_inv_std_var_dim.emplace_back(1);
    }
  }

  CudaU *mean_data = nullptr;
  CudaU *inv_var_data = nullptr;
  const bool simplified = false;

  /* Multi Output관련 정책이 정해질때까지 일단 보류
  if (!simplified) {
    Tensor* mean = ctx->Output(output_index++,
  TensorShape(mean_inv_std_var_dim)); if (mean != nullptr) { mean_data =
  reinterpret_cast<CudaU*>(mean->MutableData<U>());
    }
  }


  Tensor* var = ctx->Output(output_index, TensorShape(mean_inv_std_var_dim));
  if (var != nullptr) {
    inv_var_data = reinterpret_cast<CudaU*>(var->MutableData<U>());
  }
  */

  if (x_shape.Size() == 0) {
    return;
  }

  int device_id;
  cudaGetDevice(&device_id);

  cudaDeviceProp prop;

  cudaGetDeviceProperties(&prop, device_id);

  HostApplyLayerNorm<CudaT, CudaU, CudaV, simplified>(
      prop, stream, Y_data, mean_data, inv_var_data, X_data, n1, n2, epsilon_,
      scale_data, bias_data);
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::GetOutput() {
  cudaMemcpy(output_->data(), data_output_, output_->size(),
             cudaMemcpyDeviceToHost);
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::DeAllocateMemory() {
  cudaFree(data_output_);
  for (size_t index = 0; index < inputs_count_; index++) {
    cudaFree(data_input_[index]);
  }
}
