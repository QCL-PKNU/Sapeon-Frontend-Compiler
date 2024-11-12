#pragma once
#include "cudnn/common/cuda_common.hpp"
#include "cudnn/common/slice_compute_metadata.hpp"

namespace Cudnn {

namespace SliceCuda {

bool Impl(cudaStream_t stream, const void* input_data,
          const TensorShape& input_shape, void* output_data,
          SliceOp::PrepareForComputeMetadata& prepare_metadata,
          size_t element_size);

}  // namespace SliceCuda

class SliceBase {
  // static methods that can be used from other ops if needed
 public:
  // compute output_dims without steps (Slice V1-9 & DynamicSlice)
  static bool PrepareForCompute(
      gsl::span<const int64_t> raw_starts, gsl::span<const int64_t> raw_ends,
      gsl::span<const int64_t> raw_axes,
      SliceOp::PrepareForComputeMetadata& compute_metadata);

  // compute output_dims with steps (Slice V10)
  static bool PrepareForCompute(
      gsl::span<const int64_t> raw_starts, gsl::span<const int64_t> raw_ends,
      gsl::span<const int64_t> raw_axes, gsl::span<const int64_t> raw_steps,
      SliceOp::PrepareForComputeMetadata& compute_metadata);
};

}  // namespace Cudnn