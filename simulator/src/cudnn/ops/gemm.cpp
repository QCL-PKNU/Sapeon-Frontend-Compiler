#include "cudnn/ops/gemm.hpp"

#include "cudnn/common/cuda_common.hpp"
#include "cudnn/common/cuda_utils.hpp"
#include "cudnn/common/fpgeneric.hpp"
#include "cudnn/common/tensor_shape.hpp"
#include "gsl-lite.hpp"

#define BASE CudnnOperation
#define NAME Gemm
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

enum { NO_ERROR = 0, INVALID_ARGUMENT = -1 };

class GemmHelper {
 public:
  GemmHelper(const TensorShape& left, bool trans_left, const TensorShape& right,
             bool trans_right, const TensorShape& bias) {
    // dimension check
    assert(left.NumDimensions() == 2 || left.NumDimensions() == 1);
    assert(right.NumDimensions() == 2);

    if (trans_left) {
      M_ = left.NumDimensions() == 2 ? left[1] : left[0];
      K_ = left.NumDimensions() == 2 ? left[0] : 1;
    } else {
      M_ = left.NumDimensions() == 2 ? left[0] : 1;
      K_ = left.NumDimensions() == 2 ? left[1] : left[0];
    }

    int k_dim;
    if (trans_right) {
      N_ = right[0];
      k_dim = 1;
    } else {
      N_ = right[1];
      k_dim = 0;
    }

    if (right[k_dim] != K_) status_ = INVALID_ARGUMENT;

    if (!IsValidBroadcast(bias, M_, N_)) status_ = INVALID_ARGUMENT;

    // it is possible the input is empty tensor, for example the output of
    // roipool in fast rcnn.
    assert(M_ >= 0 && K_ > 0 && N_ >= 0);
    status_ = NO_ERROR;
  }

  int64_t M() const { return M_; }
  int64_t N() const { return N_; }
  int64_t K() const { return K_; }
  int State() const { return status_; }

 private:
  bool IsValidBroadcast(const TensorShape& bias_shape, int64_t M, int64_t N) {
    // valid shapes are (,) , (1, N) , (M, 1) , (M, N)
    if (bias_shape.NumDimensions() > 2) return false;
    // shape is (1,) or (1, 1), or (,)
    if (bias_shape.Size() == 1) return true;
    // valid bias_shape (s) are (N,) or (1, N) or (M, 1) or (M, N),
    // In last case no broadcasting needed, so don't fail it
    return ((bias_shape.NumDimensions() == 1 && bias_shape[0] == N) ||
            (bias_shape.NumDimensions() == 2 && bias_shape[0] == M &&
             (bias_shape[1] == 1 || bias_shape[1] == N)) ||
            (bias_shape.NumDimensions() == 2 && bias_shape[0] == 1 &&
             bias_shape[1] == N));
  }

 private:
  int64_t M_;
  int64_t K_;
  int64_t N_;
  int status_;
};

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

  if (layer.alpha() != layer.alpha()) {
    alpha_ = 1.0f;
  } else {
    alpha_ = layer.alpha();
  }

  if (layer.beta() != layer.beta()) {
    beta_ = 1.0f;
  } else {
    beta_ = layer.beta();
  }

  if (layer.trans_A() == std::numeric_limits<int64_t>::lowest()) {
    trans_A_ = false;
  } else {
    trans_A_ = layer.trans_A() == 0 ? false : true;
  }

  if (layer.trans_B() == std::numeric_limits<int64_t>::lowest()) {
    trans_B_ = false;
  } else {
    trans_B_ = layer.trans_B() == 0 ? false : true;
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
  std::vector<TensorShape> x_shapes;

  for (size_t index = 0; index < inputs_count_; index++) {
    std::vector<int64_t> in_vector;

    in_vector.push_back(inputs_[index]->n());

    if (inputs_[index]->c() != 1) {
      in_vector.push_back(inputs_[index]->c());
    }

    TensorShape x_shape(in_vector);
    x_shapes.push_back(x_shape);
  }

  GemmHelper helper(x_shapes[0], trans_A_, x_shapes[1], trans_B_,
                    inputs_count_ > 2 ? x_shapes[2] : TensorShape({}));

  if (helper.State() != NO_ERROR) {
    output_ = std::make_shared<Tensor>(0, 0, 0, 0, dty::GetDataType<Type>());
    return;
  }

  int M = gsl::narrow_cast<int>(helper.M());
  int N = gsl::narrow_cast<int>(helper.N());

  std::vector<int64_t> in_dims = {M, N};

  output_ = std::make_shared<Tensor>(in_dims, dty::GetDataType<Type>());
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::AllocateMemory() {
  for (size_t index = 0; index < inputs_count_; index++) {
    assert(cudaMalloc(&(data_input_[index]), inputs_[index]->size()) == 0);
    assert(cudaMemcpy(data_input_[index], inputs_[index]->data(),
                      inputs_[index]->size(), cudaMemcpyHostToDevice) == 0);
  }

  assert(cudaMalloc(&data_output_, output_->size()) == 0);
  assert(cudaMemset(data_output_, 0, output_->dimension().size()) == 0);

  int64_t count =
      output_->dimension().dims()[0] > output_->dimension().dims()[1]
          ? output_->dimension().dims()[0]
          : output_->dimension().dims()[1];

  cudaMalloc(&data_scala_, count * sizeof(Type));
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::OperationForward() {
  typedef typename ToCudaType<Type>::MappedType CudaT;

  if (output_->dimension().size() == 0) {
    return;
  }

  std::vector<TensorShape> x_shapes;

  for (size_t index = 0; index < inputs_count_; index++) {
    std::vector<int64_t> in_vector;

    in_vector.push_back(inputs_[index]->n());

    if (inputs_[index]->c() != 1) {
      in_vector.push_back(inputs_[index]->c());
    }

    TensorShape x_shape(in_vector);
    x_shapes.push_back(x_shape);
  }

  GemmHelper helper(x_shapes[0], trans_A_, x_shapes[1], trans_B_,
                    inputs_count_ > 2 ? x_shapes[2] : TensorShape({}));

  int M = gsl::narrow_cast<int>(helper.M());
  int N = gsl::narrow_cast<int>(helper.N());
  int K = gsl::narrow_cast<int>(helper.K());

  CudaT* out_data = reinterpret_cast<CudaT*>(data_output_);

  cudaStream_t stream;

  cudnnGetStream(handle_, &stream);

  int device_id;
  cudaGetDevice(&device_id);

  cudaDeviceProp device_prop;

  cudaGetDeviceProperties(&device_prop, device_id);

  cublasHandle_t cublas_handle = nullptr;

  assert(cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS);
  assert(cublasSetStream(cublas_handle, stream) == CUBLAS_STATUS_SUCCESS);

  CudaT one = ToCudaType<Type>::FromFloat(1.0f);
  CudaT zero = ToCudaType<Type>::FromFloat(0.0f);
  // broadcast bias if needed and is present
  if (beta_ != 0 && inputs_count_ > 2) {
    auto& b_shape = x_shapes[2];
    const CudaT* b_data = reinterpret_cast<const CudaT*>(data_input_[2]);

    if (b_shape.Size() == 1) {
      // if B is (), (1,) or (1, 1), broadcast the scalar
      assert(cublasCopyHelper(stream, cublas_handle, M * N, b_data, 0, out_data,
                              1) == CUBLAS_STATUS_SUCCESS);
    } else if (b_shape.NumDimensions() == 1 || b_shape[0] == 1) {
      // B is (N,) or (1, N), broadcast using Y(N,M) = 1 * B(N,1) x ones(1,M) +
      // 0 * Y
      Fill(stream, data_scala_, one, N);
      assert(cublasGemmHelper(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, 1,
                              &one, b_data, N, data_scala_, 1, &zero, out_data,
                              N, device_prop) == CUBLAS_STATUS_SUCCESS);
    } else if (b_shape.NumDimensions() == 2 && b_shape[1] == 1) {
      // B is (M, 1), broadcast using Y(N,M) = 1 * ones(N,1) x B(1,M) + 0 * Y
      Fill(stream, data_scala_, one, M);
      assert(cublasGemmHelper(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, 1,
                              &one, data_scala_, N, b_data, 1, &zero, out_data,
                              N, device_prop) == CUBLAS_STATUS_SUCCESS);
    } else {
      // B is (M, N), no broadcast needed.
      assert(cudaMemcpyAsync(out_data, b_data, M * N * sizeof(Type),
                             cudaMemcpyDeviceToDevice, stream) == 0);
      // cudaMemcpy(out_data, b_data, M * N * sizeof(Type),
      //            cudaMemcpyDeviceToDevice);
    }
  }

  CudaT alpha = ToCudaType<Type>::FromFloat(alpha_);
  CudaT beta = ToCudaType<Type>::FromFloat(beta_);
  // Gemm, note that CUDA assumes col-major, so Y(N,M) = alpha * op(W) x op(X) +
  // beta * Y
  assert(cublasGemmHelper(cublas_handle, trans_B_ ? CUBLAS_OP_T : CUBLAS_OP_N,
                          trans_A_ ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha,
                          reinterpret_cast<const CudaT*>(data_input_[1]),
                          (trans_B_ ? K : N),
                          reinterpret_cast<const CudaT*>(data_input_[0]),
                          (trans_A_ ? M : K), inputs_count_ > 2 ? &beta : &zero,
                          out_data, N, device_prop) == CUBLAS_STATUS_SUCCESS);
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
  cudaFree(data_scala_);
  for (size_t index = 0; index < inputs_count_; index++) {
    cudaFree(data_input_[index]);
  }
}
