#include "cudnn/ops/conv_base.hpp"

#include <cudnn.h>

#include <cassert>

#include "cudnn/common/conv_attributes.hpp"
#include "cudnn/common/ort_mutex.hpp"

namespace Cudnn {

cudnnStatus_t GetWorkspaceSize(
    const CudnnConvState<cudnnConvolutionFwdAlgoPerf_t>& s,
    cudnnConvolutionFwdAlgo_t algo, size_t* sz) {
  return cudnnGetConvolutionForwardWorkspaceSize(
      s.handle, s.x_tensor, s.w_desc, s.conv_desc, s.y_tensor, algo, sz);
}

size_t GetMaxWorkspaceSize(
    const CudnnConvState<cudnnConvolutionFwdAlgoPerf_t>& s,
    const cudnnConvolutionFwdAlgo_t* algo, int n_algo) {
  // TODO: get maximum available size from memory areana
  size_t free, total;
  assert(cudaMemGetInfo(&free, &total) == 0);
  // Assuming 10% of fragmentation
  free = static_cast<size_t>(static_cast<double>(free) * 0.9);
  size_t max_ws_size = 0;
  for (int i = 0; i < n_algo; i++) {
    cudnnStatus_t err;
    size_t sz;
    err = GetWorkspaceSize(s, algo[i], &sz);
    if (CUDNN_STATUS_SUCCESS != err || sz == 0 || sz < max_ws_size || sz > free)
      continue;
    max_ws_size = sz;
  }
  return max_ws_size;
}

CudnnConvolutionDescriptor::CudnnConvolutionDescriptor() : desc_(nullptr) {}

CudnnConvolutionDescriptor::~CudnnConvolutionDescriptor() {
  if (desc_ != nullptr) {
    cudnnDestroyConvolutionDescriptor(desc_);
    desc_ = nullptr;
  }
}

bool CudnnConvolutionDescriptor::Set(size_t rank,
                                     const gsl::span<const int64_t>& pads,
                                     const gsl::span<const int64_t>& strides,
                                     const gsl::span<const int64_t>& dilations,
                                     int groups, cudnnConvolutionMode_t mode,
                                     cudnnDataType_t data_type) {
  if (!desc_) assert(cudnnCreateConvolutionDescriptor(&desc_) == 0);

  InlinedVector<int, kTensorShapeSmallBufferElementsSize> pad_dims(rank);
  InlinedVector<int, kTensorShapeSmallBufferElementsSize> stride_dims(rank);
  InlinedVector<int, kTensorShapeSmallBufferElementsSize> dilation_dims(rank);
  for (size_t i = 0; i < rank; i++) {
    pad_dims[i] = gsl::narrow_cast<int>(pads[i]);
    stride_dims[i] = gsl::narrow_cast<int>(strides[i]);
    dilation_dims[i] = gsl::narrow_cast<int>(dilations[i]);
  }

  // This piece of code is copied from
  // /pytorch/aten/src/ATen/cudnn/Descriptors.h Setting math_type to
  // CUDNN_DATA_FLOAT for half input
  cudnnDataType_t math_type = data_type;
  if (data_type == CUDNN_DATA_HALF) math_type = CUDNN_DATA_FLOAT;
  assert(cudnnSetConvolutionNdDescriptor(
             desc_, gsl::narrow_cast<int>(rank), pad_dims.data(),
             stride_dims.data(), dilation_dims.data(), mode, math_type) == 0);

  assert(cudnnSetConvolutionGroupCount(desc_, groups) == 0);

  // Copied from /pytorch/aten/src/ATen/cudnn/Descriptors.h
  // See Note [behavior of cudnnFind and cudnnGet] at
  // /pytorch/aten/src/ATen/native/cudnn/Conv_v7.cpp
  assert(cudnnSetConvolutionMathType(desc_, CUDNN_DEFAULT_MATH) == 0);
  if (data_type == CUDNN_DATA_HALF) {
    assert(cudnnSetConvolutionMathType(desc_, CUDNN_TENSOR_OP_MATH) == 0);
  }

  return true;
}

}  // namespace Cudnn
