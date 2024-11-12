#include "cudnn/ops/convolution_onnx.hpp"

#define BASE CudnnOperation
#define NAME Convolution_Onnx
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
#include <cudnn.h>

#include "cudnn/common/cuda_common.hpp"
#include "cudnn/common/slice_base.hpp"
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
const cudnnConvolutionFwdAlgo_t SCOPE::kAllAlgos[] = {
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
};

template <typename Type, cudnnDataType_t DataType>
std::shared_ptr<Tensor> SCOPE::Forward(cudnnHandle_t& handle, Layer& layer) {
  input_ = layer.intermediate_activation() == nullptr
               ? layer.inputs(0)
               : layer.intermediate_activation();
  filter_ = layer.filter();
  bias_ = layer.bias();
  convolution_ = layer.convolution();
  handle_ = handle;

  InitOutputTensor();
  OperationForward();
  GetOutput();
  DeAllocateMemory();

  layer.intermediate_activation(output_);
  auto p_operation = Factory<CudnnOperation<Type>>::CreateInstance("BiasAdd");
  return p_operation.get()->Forward(handle, layer);
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::InitOutputTensor() {
  typedef typename ToCudaType<Type>::MappedType CudaT;

  s_.handle = handle_;

  // set X
  const TensorShape x_shape(input_->dimension().dims());
  const auto x_dims = x_shape.AsShapeVector();

  cudaMalloc(&data_input_, input_->size());
  cudaMemcpy(data_input_, input_->data(), input_->size(),
             cudaMemcpyHostToDevice);
  s_.x_data = reinterpret_cast<const CudaT*>(data_input_);
  s_.element_size = input_->size();

  // set W
  const TensorShape w_shape(filter_->dimension().dims());
  auto w_dims = w_shape.AsShapeVector();
  cudaMalloc(&data_filter_, filter_->size());
  cudaMemcpy(data_filter_, filter_->data(), filter_->size(),
             cudaMemcpyHostToDevice);
  s_.w_data = reinterpret_cast<const CudaT*>(data_filter_);
  // set B
  if (bias_ != nullptr) {
    if (bias_->size() > 0) {
      cudaMalloc(&data_bias_, bias_->size());
      cudaMemcpy(data_bias_, bias_->data(), bias_->size(),
                 cudaMemcpyHostToDevice);

      s_.b_data = reinterpret_cast<const CudaT*>(data_bias_);
    } else {
      s_.b_data = nullptr;
    }
  } else {
    s_.b_data = nullptr;
  }

  s_.z_data = nullptr;

  bool input_dims_changed = (s_.last_x_dims != x_dims);
  bool w_dims_changed = (s_.last_w_dims != w_dims);
  if (input_dims_changed || w_dims_changed) {
    if (input_dims_changed) s_.last_x_dims = gsl::make_span(x_dims);

    if (w_dims_changed) {
      s_.last_w_dims = gsl::make_span(w_dims);
      s_.cached_benchmark_results.clear();
    }

    const int64_t N = x_dims[0];  // X->Shape()[0];
    const int64_t M = w_dims[0];  // W->Shape()[0];

    assert(conv_attrs_.ValidateInputShape(x_shape, w_shape));

    TensorShapeVector kernel_shape;
    assert(conv_attrs_.ComputeKernelShape(w_shape, kernel_shape));

    auto rank = kernel_shape.size();
    ConvPadVector pads(conv_attrs_.pads);
    if (pads.empty()) {
      pads.resize(rank * 2, 0);
    }
    TensorShapeVector dilations(conv_attrs_.dilations);
    if (dilations.empty()) {
      dilations.resize(rank, 1);
    }
    TensorShapeVector strides(conv_attrs_.strides);
    if (strides.empty()) {
      strides.resize(rank, 1);
    }

    TensorShapeVector y_dims;
    y_dims.reserve(2 + rank);  // rank indicates number of feature dimensions -
                               // so add 2 to account for 'N' and 'C'
    y_dims.insert(y_dims.begin(), {N, M});

    TensorShapeVector y_dims_with_adjusted_pads;
    y_dims_with_adjusted_pads.reserve(
        2 + rank);  // rank indicates number of feature dimensions - so add 2 to
                    // account for 'N' and 'C'
    y_dims_with_adjusted_pads.insert(y_dims_with_adjusted_pads.begin(), {N, M});

    bool post_slicing_required = false;
    TensorShapeVector slice_starts;
    slice_starts.reserve(rank);

    TensorShapeVector slice_ends;
    slice_ends.reserve(rank);

    TensorShapeVector slice_axes;
    slice_axes.reserve(rank);

    assert(conv_attrs_.InferOutputShapeWithAdjustedPads(
        x_shape.Slice(2), kernel_shape, strides, dilations, pads, y_dims,
        y_dims_with_adjusted_pads, post_slicing_required, slice_starts,
        slice_ends, slice_axes));

    assert(y_dims.size() == y_dims_with_adjusted_pads.size());
    s_.y_dims = gsl::make_span(y_dims);
    s_.y_dims_with_adjusted_pads = y_dims_with_adjusted_pads;
    s_.post_slicing_required = post_slicing_required;
    s_.slice_starts = slice_starts;
    s_.slice_ends = slice_ends;
    s_.slice_axes = slice_axes;

    const TensorShape y_shape(s_.y_dims);

    s_.y_shape = y_shape;
    // s_.Y = context->Output(0, TensorShape(s_.y_dims));

    if (post_slicing_required) {
      // Post slicing needed. Create and fill in the Conv results in an
      // intermediate buffer.
      cudaMalloc(&data_output_, TensorShape(y_dims_with_adjusted_pads).Size() *
                                    s_.element_size);

      s_.memory_for_cudnn_conv_results = data_output_;
      s_.y_data = reinterpret_cast<CudaT*>(data_output_);
    } else {
      // No post slicing needed. Fill the output tensor's buffer directly.
      cudaMalloc(&data_output_, y_shape.Size() * sizeof(Type));
      s_.y_data = reinterpret_cast<CudaT*>(data_output_);
    }

    /*
    const CUDAExecutionProvider* cuda_ep =
        static_cast<const
    CUDAExecutionProvider*>(this->Info().GetExecutionProvider());
        */

    TensorShapeVector x_dims_cudnn{x_dims.begin(), x_dims.end()};
    TensorShapeVector y_dims_cudnn =
        !post_slicing_required ? y_dims : y_dims_with_adjusted_pads;

    if (rank < 2) {
      /*
      if (cuda_ep->GetCudnnConv1dPadToNc1d()) {
        x_dims_cudnn.insert(x_dims_cudnn.begin() + 2, 1);
        y_dims_cudnn.insert(y_dims_cudnn.begin() + 2, 1);
        w_dims.insert(w_dims.begin() + 2, 1);
        pads.insert(pads.begin() + rank, 0);
        pads.insert(pads.begin(), 0);
        kernel_shape.insert(kernel_shape.begin(), 1);
        strides.insert(strides.begin(), 1);
        dilations.insert(dilations.begin(), 1);
      } else
      */
      {
        x_dims_cudnn.push_back(1);
        y_dims_cudnn.push_back(1);
        w_dims.push_back(1);
        pads.insert(pads.begin() + rank, 0);
        pads.insert(pads.end(), 0);
        kernel_shape.push_back(1);
        strides.push_back(1);
        dilations.push_back(1);
      }
    }

    if (w_dims_changed)
      assert(s_.w_desc.Set(w_dims, CudnnTensor::GetDataType<CudaT>()));

    assert(s_.y_shape.Size() != 0);

    assert(s_.x_tensor.Set(x_dims_cudnn, CudnnTensor::GetDataType<CudaT>()));
    assert(s_.y_tensor.Set(y_dims_cudnn, CudnnTensor::GetDataType<CudaT>()));
    assert(s_.conv_desc.Set(kernel_shape.size(), pads, strides, dilations,
                            gsl::narrow_cast<int>(conv_attrs_.group),
                            CUDNN_CROSS_CORRELATION,
                            CudnnTensor::GetDataType<CudaT>()));

    if (s_.b_data != nullptr) {
      const TensorShape b_shape(bias_->dimension().dims());

      assert(b_shape.NumDimensions() == 1);
      TensorShapeVector b_dims(2 + kernel_shape.size(), 1);
      b_dims[1] = b_shape[0];
      assert(s_.b_tensor.Set(b_dims, CudnnTensor::GetDataType<CudaT>()));
    }

    if (!s_.cached_benchmark_results.contains(x_dims_cudnn)) {
      // set math type to tensor core before algorithm search
      cudnnConvolutionFwdAlgoPerf_t perf;
      int algo_count = 1;
      int cudnn_conv_algo =
          0;  // OrtCudnnConvAlgoSearchExhaustive cuda_ep->GetCudnnConvAlgo();
      assert(cudnn_conv_algo > -1 && cudnn_conv_algo < 3);

      switch (cudnn_conv_algo) {
        case 0: {
          static constexpr int num_algos = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
          size_t max_ws_size = GetMaxWorkspaceSize(s_, kAllAlgos, num_algos);

          // Use GetTransientScratchBuffer() so the workspace can be freed
          // instead of cached. Because the benchmarking uses a huge amount of
          // memory, e.g. a few GBs.
          void* algo_search_workspace;

          cudaMalloc(&algo_search_workspace, max_ws_size);
          CUDNN_RETURN_IF_ERROR(cudnnFindConvolutionForwardAlgorithmEx(
              handle_, s_.x_tensor, s_.x_data, s_.w_desc, s_.w_data,
              s_.conv_desc, s_.y_tensor, s_.y_data,
              1,            // requestedAlgoCount
              &algo_count,  // returnedAlgoCount
              &perf, algo_search_workspace, max_ws_size));
          break;
        }
        case 1:
          assert(cudnnGetConvolutionForwardAlgorithm_v7(
              handle_, s_.x_tensor, s_.w_desc, s_.conv_desc, s_.y_tensor,
              1,            // requestedAlgoCount
              &algo_count,  // returnedAlgoCount
              &perf));
          break;

        default:
          perf.algo = kDefaultConvAlgo;
          assert(GetWorkspaceSize(s_, perf.algo, &perf.memory));
          perf.mathType = CUDNN_DEFAULT_MATH;
      }
      s_.cached_benchmark_results.insert(
          x_dims_cudnn, {perf.algo, perf.memory, perf.mathType});
    }

    const auto& perf = s_.cached_benchmark_results.at(x_dims_cudnn);
    CUDNN_RETURN_IF_ERROR(
        cudnnSetConvolutionMathType(s_.conv_desc, perf.mathType));
    s_.algo = perf.algo;
    s_.workspace_bytes = perf.memory;
  } else {
    // set Y
    const TensorShape y_shape(s_.y_dims);

    s_.y_shape = y_shape;
    assert(y_shape.Size() != 0);

    if (s_.post_slicing_required) {
      // Post slicing needed. Create and fill in the Conv results in an
      // intermediate buffer.
      cudaMalloc(
          &data_output_,
          TensorShape(s_.y_dims_with_adjusted_pads).Size() * s_.element_size);

      s_.memory_for_cudnn_conv_results = data_output_;
      s_.y_data = reinterpret_cast<CudaT*>(data_output_);
    } else {
      // No post slicing needed. Fill the output tensor's buffer directly.
      cudaMalloc(&data_output_, y_shape.Size() * sizeof(Type));
      s_.y_data = reinterpret_cast<CudaT*>(data_output_);
    }
  }

  std::vector<int64_t> y_dims;

  for (int i = 0; i < s_.y_shape.Size(); i++) {
    y_dims.push_back(s_.y_dims[i]);
  }

  output_ = std::make_shared<Tensor>(y_dims, dty::GetDataType<Type>());
}

bool SliceOutUnwantedOutputSection(cudaStream_t stream, const void* input_data,
                                   gsl::span<const int64_t> input_dims,
                                   void* output_data,
                                   const gsl::span<const int64_t>& output_dims,
                                   const gsl::span<const int64_t>& starts,
                                   const gsl::span<const int64_t>& ends,
                                   const gsl::span<const int64_t>& axes,
                                   size_t element_size) {
  Cudnn::SliceOp::PrepareForComputeMetadata compute_metadata(input_dims);

  assert(Cudnn::SliceBase::PrepareForCompute(starts, ends, axes,
                                             compute_metadata));

  // As a sanity check, ensure that the slice operator's output shape matches
  // with the expected output shape
  auto a = gsl::make_span(compute_metadata.output_dims_);
  auto b = output_dims;

  assert(std::equal(a.begin(), a.end(), b.begin(), b.end()));

  return Cudnn::SliceCuda::Impl(stream, input_data, input_dims, output_data,
                                compute_metadata, element_size);
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::OperationForward() {
  typedef typename ToCudaType<Type>::MappedType CudaT;

  const auto alpha = Consts<CudaT>::One;
  const auto beta = Consts<CudaT>::Zero;

  void* workspace;
  cudaMalloc(&workspace, s_.workspace_bytes);

  CUDNN_RETURN_IF_ERROR(cudnnConvolutionForward(
      handle_, &alpha, s_.x_tensor, s_.x_data, s_.w_desc, s_.w_data,
      s_.conv_desc, s_.algo, workspace, s_.workspace_bytes, &beta, s_.y_tensor,
      s_.y_data));

  if (nullptr != s_.b_data) {
    CUDNN_RETURN_IF_ERROR(cudnnAddTensor(handle_, &alpha, s_.b_tensor,
                                         s_.b_data, &alpha, s_.y_tensor,
                                         s_.y_data));
  }

  cudaStream_t stream;
  cudnnGetStream(handle_, &stream);

  if (s_.post_slicing_required) {
    assert(SliceOutUnwantedOutputSection(
               stream, s_.y_data, gsl::make_span(s_.y_dims_with_adjusted_pads),
               s_.y_data, s_.y_dims.GetDims(), s_.slice_starts, s_.slice_ends,
               s_.slice_axes, s_.element_size) == true);
  }
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::GetOutput() {
  cudaMemcpy(output_->data(), data_output_, output_->size(),
             cudaMemcpyDeviceToHost);
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::DeAllocateMemory() {
  cudaFree(data_workspace_);
  cudaFree(data_filter_);
  cudaFree(data_output_);
  cudaFree(data_input_);
}
