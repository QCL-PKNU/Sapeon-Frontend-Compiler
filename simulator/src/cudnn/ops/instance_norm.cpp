#include "cudnn/ops/instance_norm.hpp"

#include "cudnn/common/common.cuh"
#include "cudnn/common/cuda_common.hpp"
#include "cudnn/common/cudnn_common.hpp"
#include "cudnn/common/tensor_shape.hpp"
#include "cudnn/ops/batch_norm_helper.hpp"
#include "cudnn/ops/instance_norm_helper.hpp"
#include "cudnn/ops/instance_norm_impl.cuh"
#include "gsl-lite.hpp"

#define BASE CudnnOperation
#define NAME InstanceNorm
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

template <typename Type, cudnnDataType_t DataType>
shared_ptr<Tensor> SCOPE::Forward(cudnnHandle_t& handle, Layer& layer) {
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

  std::vector<std::vector<int64_t>> in_vectors;

  for (size_t index = 0; index < 3; index++) {
    in_vectors.push_back(inputs_[index]->dimension().dims());
  }

  const TensorShape x_shape(in_vectors[0]);
  const TensorShape scale_shape(in_vectors[1]);
  const TensorShape bias_shape(in_vectors[2]);

  assert(
      InstanceNormHelper::ValidateInputs(&x_shape, &scale_shape, &bias_shape));

  auto* y_data = reinterpret_cast<CudaT*>(data_output_);
  const auto* x_data = reinterpret_cast<const CudaT*>(data_input_[0]);
  const auto* scale_data = reinterpret_cast<const CudaT*>(data_input_[1]);
  const auto* bias_data = reinterpret_cast<const CudaT*>(data_input_[2]);

  const auto& x_dims = x_shape.GetDims();
  const int64_t N = x_dims[0];
  const int64_t C = x_dims[1];
  const auto one = Consts<CudaT>::One;
  const auto zero = Consts<CudaT>::Zero;

  if (N == 1) {
    // when N == 1, we can treat it as spatial batch normalization in training
    // as the mean/variance would be computed from input

    CudnnTensor data_desc;
    std::vector<int64_t> new_dims;
    BatchNormHelper::NormalizeDims(x_shape, new_dims);
    assert(data_desc.Set(new_dims, CudnnTensor::GetDataType<CudaT>()));

    CudnnTensor stats_desc;
    assert(stats_desc.Set(data_desc, CUDNN_BATCHNORM_SPATIAL));

    CUDNN_RETURN_IF_ERROR(BatchNormalizationForwardTrainingHelper(
        handle_, CUDNN_BATCHNORM_SPATIAL, &one, &zero, data_desc, x_data,
        data_desc, y_data, stats_desc, scale_data, bias_data, 1.0f, nullptr,
        nullptr, epsilon_, nullptr, nullptr));
  } else {
    // we use cudnnBatchNormalizationForwardTraining to compute mean/variance
    // so collapsing NC into channel

    auto input_count = x_shape.Size();              // N * C * H * W
    auto stats_count = x_shape.SizeToDimension(2);  // N * C
    auto image_size = input_count / stats_count;

    CudnnTensor data_desc;
    assert(data_desc.Set(std::array<int64_t, 4>{1, stats_count, image_size, 1},
                         CudnnTensor::GetDataType<CudaT>()));

    CudnnTensor stats_desc;
    assert(stats_desc.Set(std::array<int64_t, 4>{1, stats_count, 1, 1},
                          CudnnTensor::GetDataType<CudaT>()));

    const size_t stats_byte_count = stats_count * sizeof(CudaT);

    // Mean & Variance are inputs & outputs and must be initialized to zero to
    // work properly
    CudaT *mean, *variance, *unused_scale, *unused_bias;

    cudaMalloc(&mean, stats_count * sizeof(CudaT));

    assert(cudaMemsetAsync(mean, 0, stats_byte_count, stream) == cudaSuccess);

    cudaMalloc(&variance, stats_count * sizeof(CudaT));

    assert(cudaMemsetAsync(variance, 0, stats_byte_count, stream) ==
           cudaSuccess);

    cudaMalloc(&unused_scale, stats_count * sizeof(CudaT));

    assert(cudaMemsetAsync(unused_scale, 0, stats_byte_count, stream) ==
           cudaSuccess);

    cudaMalloc(&unused_bias, stats_count * sizeof(CudaT));
    assert(cudaMemsetAsync(unused_bias, 0, stats_byte_count, stream) ==
           cudaSuccess);

    // first, compute mean and variance per-instance per-channel using
    // cudnnBatchNorm training
    CUDNN_RETURN_IF_ERROR(BatchNormalizationForwardTrainingHelper(
        handle_, CUDNN_BATCHNORM_SPATIAL, &one, &zero, data_desc, x_data,
        data_desc,
        y_data,  // use y temporarily, would be rewritten later
        stats_desc, unused_scale, unused_bias, 1.0f, mean, variance,
        CUDNN_BN_MIN_EPSILON, nullptr, nullptr));

    // Y = scale * (x - mean) / sqrt (variance + epsilon) + B
    // X/Y is (N,C,H,W)
    // scale/bias is (1,C,1,1)
    // mean/stddev is (N,C,1,1)
    // NOTE cudnnBatchNormalization computes unbiased variance sum((Xi -
    // mean)^2) / (count - 1) and it needs to be corrected with (count - 1) /
    // count
    fast_divmod fdm_HW(gsl::narrow_cast<int>(image_size));
    fast_divmod fdm_C(gsl::narrow_cast<int>(C));

    InstanceNormImpl<CudaT>(stream, x_data, scale_data, bias_data, mean,
                            variance, (image_size - 1.0) / image_size,
                            static_cast<double>(epsilon_), fdm_HW, fdm_C,
                            y_data, input_count);
    cudaFree(mean);
    cudaFree(variance);
    cudaFree(unused_scale);
    cudaFree(unused_bias);
  }
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
