#include "gsl-lite.hpp"

#define NONE
#include "cudnn/common/common.cuh"
#include "cudnn/common/conv_transpose_attributes.hpp"
#include "cudnn/common/cuda_common.hpp"
#include "cudnn/common/cuda_utils.hpp"
#include "cudnn/common/cudnn_common.hpp"
#include "cudnn/ops/conv_transpose.hpp"
#include "glog/logging.h"

#define BASE CudnnOperation
#define NAME ConvTranspose
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

  SetOptions(layer);

  OperationForward();
  GetOutput();
  DeAllocateMemory();

  return output_;
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::SetOptions(Layer& layer) {
  std::string auto_pad_str;
  int64_t in_group;

  TensorShapeVector kernel_shape = ToShapeVector(layer.kernel_shape());
  TensorShapeVector in_strides = ToShapeVector(layer.strides());
  TensorShapeVector in_dilations = ToShapeVector(layer.dilations());
  TensorShapeVector in_output_padding = ToShapeVector(layer.output_padding());
  TensorShapeVector in_output_shape = ToShapeVector(layer.output_shape());
  TensorShapeVector pads_span = ToShapeVector(layer.pads());

  auto_pad_str = layer.auto_pad();

  if (layer.group() == std::numeric_limits<int64_t>::lowest()) {
    in_group = 1;
  } else {
    in_group = layer.group();
  }

  conv_transpose_attrs_.SetConvTransposeAttributes(
      auto_pad_str, kernel_shape, in_strides, ToConstSpan(pads_span),
      in_dilations, in_group, in_output_padding, in_output_shape);
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
  bool dynamic_padding = false;

  typedef typename ToCudaType<Type>::MappedType CudaT;

  const TensorShape x_shape(inputs_[0]->dimension().dims());
  auto x_dims = x_shape.AsShapeVector();
  // auto x_data = reinterpret_cast<const CudaT*>(X->Data<T>());

  auto x_dimensions = x_shape.NumDimensions();
  std::string message;
  if (x_dimensions < 3 || x_dimensions > 5) {
    // TODO: the error message should tell which operator raises it.
    LOG(ERROR) << "Input X must be 3-, 4- or 5-dimensional."
               << " X: " << x_shape.ToString();
    assert(false);
  }

  const TensorShape w_shape(inputs_[1]->dimension().dims());

  TensorShapeVector w_dims = w_shape.AsShapeVector();
  // auto w_data = reinterpret_cast<const CudaT*>(W->Data<T>());

  TensorShape p_shape;
  TensorShape b_shape;

  size_t num_inputs = inputs_count_;
  bool has_bias = dynamic_padding ? num_inputs == 4 : num_inputs == 3;

  CudaT* y_data = nullptr;
  if (x_dimensions == 3) {
    x_dims.insert(x_dims.begin() + 2, 1);
    w_dims.insert(w_dims.begin() + 2, 1);
  }

  {
    std::lock_guard<OrtMutex> lock(s_.mutex);
    // TODO: add a global cache if need to handle cases for multiple frames
    // running simultaneously with different batch_size
    bool input_dims_changed = (s_.last_x_dims.AsShapeVector() != x_dims);
    bool w_dims_changed = (s_.last_w_dims.AsShapeVector() != w_dims);
    if (input_dims_changed || w_dims_changed) {
      if (input_dims_changed) s_.last_x_dims = gsl::make_span(x_dims);

      if (w_dims_changed) {
        s_.last_w_dims = gsl::make_span(w_dims);
        s_.cached_benchmark_results.clear();
      }

      if (has_bias) {
        b_shape = dynamic_padding ? inputs_[3]->dimension().dims()
                                  : inputs_[2]->dimension().dims();
      }

      ConvTransposeAttributes::Prepare p;
      p.x_shape = x_shape;
      p.f_shape = w_shape;

      std::vector<int64_t> pads_data;

      assert(conv_transpose_attrs_.PrepareForCompute(p, p_shape, pads_data,
                                                     dynamic_padding));

      auto y_dims = p.y_shape.AsShapeVector();

      output_ = std::make_shared<Tensor>(y_dims, dty::GetDataType<Type>());
      if (x_dimensions == 3) {
        y_dims.insert(y_dims.begin() + 2, 1);
        p.kernel_shape.insert(p.kernel_shape.begin(), 1);
        p.pads.insert(p.pads.begin(), 0);
        p.pads.insert(p.pads.begin() + 2, 0);
        p.strides.insert(p.strides.begin(), 1);
        p.dilations.insert(p.dilations.begin(), 1);
      }
      s_.y_dims = gsl::make_span(y_dims);

      if (w_dims_changed)
        assert(s_.w_desc.Set(w_dims, CudnnTensor::GetDataType<CudaT>()));

      if (p.y_shape.Size() == 0) {
        return;
      }

      AllocateMemory();

      assert(s_.x_tensor.Set(x_dims, CudnnTensor::GetDataType<CudaT>()));
      assert(s_.y_tensor.Set(y_dims, CudnnTensor::GetDataType<CudaT>()));

      cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;

      assert(s_.conv_desc.Set(
          p.kernel_shape.size(), p.pads, p.strides, p.dilations,
          gsl::narrow_cast<int>(conv_transpose_attrs_.group), mode,
          CudnnTensor::GetDataType<CudaT>()));

      if (has_bias) {
        const auto& b_shape = p.b_shape;
        assert(b_shape.NumDimensions() == 1);
        TensorShapeVector b_dims(2 + p.kernel_shape.size());
        b_dims[0] = 1;           // N
        b_dims[1] = b_shape[0];  // C
        for (size_t i = 0; i < p.kernel_shape.size(); i++) b_dims[2 + i] = 1;

        assert(s_.b_tensor.Set(b_dims, CudnnTensor::GetDataType<CudaT>()));
      }

      y_data = reinterpret_cast<CudaT*>(data_output_);
      CudaT* x_data = reinterpret_cast<CudaT*>(data_input_[0]);
      CudaT* w_data = reinterpret_cast<CudaT*>(data_input_[1]);
      CudaT* b_data =
          has_bias
              ? (dynamic_padding ? reinterpret_cast<CudaT*>(data_input_[3])
                                 : reinterpret_cast<CudaT*>(data_input_[2]))
              : nullptr;

      if (!s_.cached_benchmark_results.contains(x_dims)) {
        void* algo_search_workspace;

        cudaMalloc(&algo_search_workspace, AlgoSearchWorkspaceSize);

        // set math type to tensor core before algorithm search
        /*
        if constexpr (std::is_same<T, MLFloat16>::value)
          CUDNN_RETURN_IF_ERROR(
              cudnnSetConvolutionMathType(s_.conv_desc, CUDNN_TENSOR_OP_MATH));
              */

        cudnnConvolutionBwdDataAlgoPerf_t perf;
        int algo_count = 1;
        assert(cudnnFindConvolutionBackwardDataAlgorithmEx(
                   handle_, s_.w_desc, w_data, s_.x_tensor, x_data,
                   s_.conv_desc, s_.y_tensor, y_data, 1, &algo_count, &perf,
                   algo_search_workspace, AlgoSearchWorkspaceSize) == 0);
        s_.cached_benchmark_results.insert(
            x_dims, {perf.algo, perf.memory, perf.mathType});
        cudaFree(algo_search_workspace);
      }

      const auto& perf = s_.cached_benchmark_results.at(x_dims);
      assert(cudnnSetConvolutionMathType(s_.conv_desc, perf.mathType) == 0);
      s_.algo = perf.algo;
      s_.workspace_bytes = perf.memory;
    }
    // The following block will be executed in case there has been no change in
    // the shapes of the input and the filter compared to the previous run
    if (!y_data) {
      auto y_dims = s_.y_dims.AsShapeVector();

      if (x_dimensions == 3) {
        y_dims.erase(y_dims.begin() + 2);
      }

      output_ = std::make_shared<Tensor>(y_dims, dty::GetDataType<Type>());

      AllocateMemory();

      y_data = reinterpret_cast<CudaT*>(data_output_);

      // Bail out early if one of the output dimensions is zero.
      if (y_dims.size() == 0) {
        return;
      }
    }

    const auto alpha = Consts<CudaT>::One;
    const auto beta = Consts<CudaT>::Zero;

    CudaT* x_data = reinterpret_cast<CudaT*>(data_input_[0]);
    CudaT* w_data = reinterpret_cast<CudaT*>(data_input_[1]);

    void* workspace;

    cudaMalloc(&workspace, s_.workspace_bytes);

    assert(cudnnConvolutionBackwardData(handle_, &alpha, s_.w_desc, w_data,
                                        s_.x_tensor, x_data, s_.conv_desc,
                                        s_.algo, workspace, s_.workspace_bytes,
                                        &beta, s_.y_tensor, y_data) == 0);

    if (has_bias) {
      auto b_data = dynamic_padding ? reinterpret_cast<CudaT*>(data_input_[3])
                                    : reinterpret_cast<CudaT*>(data_input_[2]);
      assert(cudnnAddTensor(handle_, &alpha, s_.b_tensor, b_data, &alpha,
                            s_.y_tensor, y_data) == 0);
    }

    cudaFree(&workspace);
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
  for (size_t index = 0; index < inputs_count_; index++) {
    cudaFree(data_input_[index]);
  }
}

}  // namespace Cudnn
