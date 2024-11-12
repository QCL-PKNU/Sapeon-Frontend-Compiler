// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CUDNN_COMMON_REDUCTION_OPS_HPP
#define CUDNN_COMMON_REDUCTION_OPS_HPP

#include <cassert>

#include "cudnn/common/reduction_functions.hpp"
#include "cudnn/common/reduction_ops.hpp"
#include "cudnn/common/tensor_shape.hpp"

namespace Cudnn {

// Holds some metadata that will be used during actual reduction op compute time
struct PrepareReduceMetadata {
  int64_t input_count;
  int64_t output_count;
  // This holds the output dims without any reduced dims squeezed (even if
  // keep_dims == 1)
  TensorShapeVector output_dims;
  // This holds the output dims with with reduced dims squeezed (if keep_dims ==
  // 1)
  TensorShapeVector squeezed_output_dims;
  TensorShapeVector input_dims_cudnn;
  TensorShapeVector output_dims_cudnn;
};

bool PrepareForReduce(TensorShape x_shape, bool keepdims,
                      gsl::span<const int64_t> axes,
                      PrepareReduceMetadata& prepare_reduce_metadata,
                      const TensorShape* input_shape_override = nullptr);

template <typename T, cudnnReduceTensorIndices_t ReduceTensorIndices>
bool ReduceComputeCore(cudaStream_t stream, cudnnHandle_t cudnn_handle,
                       const TensorShape& x_shape, T* x_data,
                       PrepareReduceMetadata& prepare_reduce_metadata,
                       TensorShape& output_shape, T* output_data,
                       cudnnReduceTensorOp_t cudnn_reduce_op,
                       gsl::span<const int64_t> axes, bool calculate_log,
                       bool calculate_sqt, bool log_sum_exp,
                       bool fast_reduction, int out_type,
                       const TensorShape* input_shape_override = nullptr);

class CudnnReduceDescriptor final {
 public:
  CudnnReduceDescriptor() : desc_(nullptr) {}

  ~CudnnReduceDescriptor() {
    if (desc_ != nullptr) {
      cudnnDestroyReduceTensorDescriptor(desc_);
      desc_ = nullptr;
    }
  }

  CudnnReduceDescriptor(const CudnnReduceDescriptor&) = delete;
  CudnnReduceDescriptor& operator=(const CudnnReduceDescriptor&) = delete;

  bool Set(cudnnReduceTensorOp_t op, cudnnDataType_t type,
           cudnnReduceTensorIndices_t indices) {
    if (!desc_) assert(cudnnCreateReduceTensorDescriptor(&desc_) == 0);

    assert(cudnnSetReduceTensorDescriptor(
               desc_, op, type, CUDNN_PROPAGATE_NAN, indices,
               CUDNN_32BIT_INDICES) == 0);  // currently only the 32-bit
                                            // (unsigned int) type is supported.
    return true;
  }

  operator cudnnReduceTensorDescriptor_t() const { return desc_; }

 private:
  cudnnReduceTensorDescriptor_t desc_;
};

}  // namespace Cudnn

#endif  // CUDNN_COMMON_REDUCTION_OPS_HPP
