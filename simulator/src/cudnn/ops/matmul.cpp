#include "cudnn/ops/matmul.hpp"

#include "cudnn/common/cuda_common.hpp"
#include "cudnn/common/fpgeneric.hpp"
#include "cudnn/common/matmul_helper.hpp"
#include "gsl-lite.hpp"

#define BASE CudnnOperation
#define NAME MatMul
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

#include <cassert>
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

// StridedBatchedGemm can be used for the following GEMM computation
// C[pnm] = A[pnk]*B[km] or C[pnm] = A[pnk]*B[pkm]
static bool CanUseStridedBatchedGemm(const TensorShape& left_shape,
                                     const TensorShape& right_shape,
                                     bool transa, bool transb,
                                     bool trans_batch_a, bool trans_batch_b,
                                     int64_t& stride_A, int64_t& stride_B,
                                     int64_t& stride_C, int64_t& batch_count) {
  size_t left_num_dims = left_shape.NumDimensions();
  size_t right_num_dims = right_shape.NumDimensions();

  if (!(left_num_dims >= 3 && right_num_dims >= 2)) {
    return false;
  }

  size_t left_leading_axis = trans_batch_a ? 0 : left_num_dims - 2;
  size_t right_leading_axis = trans_batch_b ? 0 : right_num_dims - 2;
  int64_t left_p = left_shape.SizeToDimension(left_num_dims - 2);
  if (trans_batch_a) {
    left_p = left_p * left_shape[left_num_dims - 2] / left_shape[0];
  }
  int64_t left_k =
      transa ? left_shape[left_leading_axis] : left_shape[left_num_dims - 1];

  if (right_num_dims >= 3) {
    int64_t right_p = right_shape.SizeToDimension(right_num_dims - 2);
    if (trans_batch_b) {
      right_p = right_p * right_shape[right_num_dims - 2] / right_shape[0];
    }
    if (left_p != right_p) {
      return false;
    }
  }

  int64_t right_k = transb ? right_shape[right_num_dims - 1]
                           : right_shape[right_leading_axis];
  if (left_k != right_k) {
    return false;
  }

  int64_t n =
      transa ? left_shape[left_num_dims - 1] : left_shape[left_leading_axis];
  int64_t m = transb ? right_shape[right_leading_axis]
                     : right_shape[right_num_dims - 1];
  stride_A = n * left_k / (trans_batch_a ? left_shape[0] : 1);
  stride_B = right_num_dims == 2
                 ? 0
                 : right_k * m / (trans_batch_b ? right_shape[0] : 1);
  stride_C = n * m;
  batch_count = left_p;
  return true;
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

  alpha_ = 1.0f;
  trans_A_ = false;
  trans_B_ = false;
  trans_batch_a_ = false;
  trans_batch_b_ = false;

  InitOutputTensor();
  AllocateMemory();
  OperationForward();
  GetOutput();
  DeAllocateMemory();

  return output_;
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::InitOutputTensor() {
  MatMulComputeHelper helper;
  std::vector<TensorShape> x_shapes;

  for (size_t index = 0; index < inputs_count_; index++) {
    std::vector<int64_t> in_vector = inputs_[index]->dimension().dims();
    TensorShape x_shape(in_vector);
    x_shapes.push_back(x_shape);
  }

  if (x_shapes[0].NumDimensions() == 1) {
    trans_A_ = false;
  }
  if (x_shapes[1].NumDimensions() == 1) {
    trans_B_ = false;
  }

  assert(helper.Compute(x_shapes[0], x_shapes[1], trans_A_, trans_B_,
                        trans_batch_a_, trans_batch_b_, false));

  TensorShape output_shape = helper.OutputShape();

  output_ = std::make_shared<Tensor>(output_shape.AsShapeVector(),
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
  cudaMemset(data_output_, 0, output_->size());
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::OperationForward() {
  typedef typename ToCudaType<Type>::MappedType CudaT;

  std::vector<TensorShape> x_shapes;

  for (size_t index = 0; index < inputs_count_; index++) {
    std::vector<int64_t> in_vector = inputs_[index]->dimension().dims();
    TensorShape x_shape(in_vector);
    x_shapes.push_back(x_shape);
  }

  MatMulComputeHelper helper;

  assert(helper.Compute(x_shapes[0], x_shapes[1], trans_A_, trans_B_,
                        trans_batch_a_, trans_batch_b_, false));

  TensorShape output_shape = helper.OutputShape();

  if (output_shape.Size() == 0) {
    return;
  }

  cudaStream_t stream;

  cudnnGetStream(handle_, &stream);

  const CudaT alpha = ToCudaType<Type>::FromFloat(alpha_);
  const CudaT zero = ToCudaType<Type>::FromFloat(0.0f);

  cublasOperation_t transA = trans_A_ ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = trans_B_ ? CUBLAS_OP_T : CUBLAS_OP_N;
  const int lda = helper.Lda(trans_A_);
  const int ldb = helper.Ldb(trans_B_);
  const int ldc = helper.Ldc();
  int64_t stride_A, stride_B, stride_C, batch_count;

  int device_id;
  cudaGetDevice(&device_id);

  cudaDeviceProp device_prop;

  cudaGetDeviceProperties(&device_prop, device_id);

  cublasHandle_t cublas_handle = nullptr;

  assert(cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS);
  assert(cublasSetStream(cublas_handle, stream) == CUBLAS_STATUS_SUCCESS);

  if (helper.OutputOffsets().size() == 1) {
    assert(cublasGemmHelper(
               cublas_handle, transB, transA, static_cast<int>(helper.N()),
               static_cast<int>(helper.M()), static_cast<int>(helper.K()),
               &alpha, reinterpret_cast<const CudaT*>(data_input_[1]), ldb,
               reinterpret_cast<const CudaT*>(data_input_[0]), lda, &zero,
               reinterpret_cast<CudaT*>(data_output_), ldc,
               device_prop) == CUBLAS_STATUS_SUCCESS);
    return;
  } else if (CanUseStridedBatchedGemm(
                 x_shapes[0], x_shapes[1], trans_A_, trans_B_, trans_batch_a_,
                 trans_batch_b_, stride_A, stride_B, stride_C, batch_count)) {
    cublasStatus_t status = cublasGemmStridedBatchedHelper(
        cublas_handle, transB, transA, static_cast<int>(helper.N()),
        static_cast<int>(helper.M()), static_cast<int>(helper.K()), &alpha,
        reinterpret_cast<const CudaT*>(data_input_[1]), ldb, stride_B,
        reinterpret_cast<const CudaT*>(data_input_[0]), lda, stride_A, &zero,
        reinterpret_cast<CudaT*>(data_output_), ldc, stride_C,
        static_cast<int>(batch_count), device_prop);
    assert(status == CUBLAS_STATUS_SUCCESS);
    return;
  }

  helper.FillOffsets();

  size_t data_offset_left_size = helper.LeftOffsets().size() * sizeof(CudaT*);
  size_t data_offset_right_size = helper.RightOffsets().size() * sizeof(CudaT*);
  size_t data_offset_output_size =
      helper.OutputOffsets().size() * sizeof(CudaT*);

  CudaT** left_arrays = (CudaT**)malloc(data_offset_left_size);
  CudaT** right_arrays = (CudaT**)malloc(data_offset_right_size);
  CudaT** output_arrays = (CudaT**)malloc(data_offset_output_size);

  std::memset(left_arrays, 0, data_offset_left_size);
  std::memset(right_arrays, 0, data_offset_right_size);
  std::memset(output_arrays, 0, data_offset_output_size);

  MatMulComputeHelper::OffsetToArrays(
      reinterpret_cast<CudaT*>(data_input_[0]), helper.LeftOffsets(),
      gsl::span<CudaT*>(left_arrays, helper.LeftOffsets().size()));
  MatMulComputeHelper::OffsetToArrays(
      reinterpret_cast<CudaT*>(data_input_[1]), helper.RightOffsets(),
      gsl::span<CudaT*>(right_arrays, helper.RightOffsets().size()));
  MatMulComputeHelper::OffsetToArrays(
      reinterpret_cast<CudaT*>(data_output_), helper.OutputOffsets(),
      gsl::span<CudaT*>(output_arrays, helper.OutputOffsets().size()));

  Type** data_offset_left;
  Type** data_offset_right;
  Type** data_offset_output;

  cudaMalloc(&data_offset_left, data_offset_left_size);
  cudaMemcpy(data_offset_left, left_arrays, data_offset_left_size,
             cudaMemcpyHostToDevice);

  cudaMalloc(&data_offset_right, data_offset_right_size);
  cudaMemcpy(data_offset_right, right_arrays, data_offset_right_size,
             cudaMemcpyHostToDevice);

  cudaMalloc(&data_offset_output, data_offset_output_size);
  cudaMemcpy(data_offset_output, output_arrays, data_offset_right_size,
             cudaMemcpyHostToDevice);

  free(left_arrays);
  free(right_arrays);
  free(output_arrays);

  assert(cublasGemmBatchedHelper(
             cublas_handle, transB, transA, static_cast<int>(helper.N()),
             static_cast<int>(helper.M()), static_cast<int>(helper.K()), &alpha,
             (const Type**)data_offset_right, ldb,
             (const Type**)data_offset_left, lda, &zero, data_offset_output,
             ldc, static_cast<int>(helper.OutputOffsets().size()),
             device_prop) == 0);

  cudaFree(data_offset_left);
  cudaFree(data_offset_right);
  cudaFree(data_offset_output);
  cublasDestroy(cublas_handle);
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
