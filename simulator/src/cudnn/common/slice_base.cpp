#include "cudnn/common/slice_base.hpp"

#include "cudnn/common/cuda_utils.hpp"
#include "cudnn/common/slice_helper.hpp"
#include "cudnn/common/slice_impl.cuh"
#include "cudnn/common/tensor_shape.hpp"
#include "cudnn/common/utils.hpp"

namespace Cudnn {

static bool SliceImpCore(cudaStream_t stream, const void* input_data,
                         void* output_data, size_t element_size,
                         size_t dimension_count,
                         const TArray<int64_t>& starts_buffer,
                         const TArray<int64_t>& steps_buffer,
                         const TArray<int64_t>& input_strides,
                         const TArray<fast_divmod>& output_strides,
                         const TensorShape& output_shape) {
  if (output_shape.Size() == 0) {
    return true;
  }

  return SliceImpl(stream, element_size,
                   gsl::narrow_cast<int32_t>(dimension_count), starts_buffer,
                   steps_buffer, input_strides, output_strides, input_data,
                   output_data, output_shape.Size());
}

namespace SliceCuda {

static bool ComputeSliceStrides(
    const TensorShape& input_shape, TArray<int64_t>& input_strides,
    TArray<fast_divmod>& output_strides,
    SliceOp::PrepareForComputeMetadata& compute_metadata) {
  // If we were able to coalesce the input and output shapes, use the new shapes
  // to compute the strides.
  const auto input_dimensions = input_shape.GetDims();
  size_t rank = compute_metadata.p_flattened_input_dims_
                    ? compute_metadata.p_flattened_input_dims_->size()
                    : input_dimensions.size();
  input_strides.SetSize(gsl::narrow_cast<int32_t>(rank));
  const gsl::span<int64_t> input_strides_span =
      gsl::make_span(input_strides.Data(), input_strides.Size());
  if (compute_metadata.p_flattened_input_dims_) {
    assert(TensorPitches::Calculate(input_strides_span,
                                    compute_metadata.flattened_input_dims_));
  } else {
    assert(TensorPitches::Calculate(input_strides_span, input_dimensions));
  }

  const auto output_dims =
      gsl::make_span(compute_metadata.p_flattened_output_dims_ != nullptr
                         ? compute_metadata.flattened_output_dims_
                         : compute_metadata.output_dims_);
  TensorPitches original_output_strides(output_dims);
  output_strides.SetSize(
      gsl::narrow_cast<int32_t>(original_output_strides.size()));
  for (int32_t i = 0,
               limit = static_cast<int32_t>(original_output_strides.size());
       i < limit; ++i) {
    output_strides[i] =
        fast_divmod(gsl::narrow_cast<int>(original_output_strides[i]));
  }

  return true;
}

bool Impl(cudaStream_t stream, const void* input_data,
          const TensorShape& input_shape, void* output_data,
          SliceOp::PrepareForComputeMetadata& compute_metadata,
          size_t element_size) {
  const auto input_dimensions = input_shape.GetDims();
  size_t dimension_count = input_dimensions.size();

  TArray<int64_t> starts_buffer(compute_metadata.starts_);
  TArray<int64_t> steps_buffer(compute_metadata.steps_);
  TArray<int64_t> input_strides;
  TArray<fast_divmod> output_strides;

  assert(ComputeSliceStrides(input_shape, input_strides, output_strides,
                             compute_metadata));

  TensorShape output_shape(compute_metadata.output_dims_);

  assert(SliceImpCore(stream, input_data, output_data, element_size,
                      gsl::narrow_cast<int32_t>(dimension_count), starts_buffer,
                      steps_buffer, input_strides, output_strides,
                      output_shape));

  return true;
}

}  // namespace SliceCuda

static void FlattenOutputDims(gsl::span<const int64_t> input_dimensions,
                              gsl::span<const int64_t> output_dims,
                              TensorShapeVector& starts,
                              TensorShapeVector& ends, TensorShapeVector& steps,
                              TensorShapeVector*& p_flattened_input_dims,
                              TensorShapeVector*& p_flattened_output_dims) {
  size_t cur = 0;
  size_t nxt = 0;
  while (true) {
    // Skip all leading slicing dims.
    while (nxt < starts.size() &&
           (steps[nxt] != 1 || input_dimensions[nxt] != output_dims[nxt])) {
      p_flattened_input_dims->emplace_back(input_dimensions[nxt]);
      p_flattened_output_dims->emplace_back(output_dims[nxt]);
      starts[cur] = starts[nxt];
      ends[cur] = ends[nxt];
      steps[cur] = steps[nxt];
      ++cur;
      ++nxt;
    }
    if (nxt == starts.size()) {
      break;
    }
    // Coalesce contiguous non-slicing dims.
    int64_t running_size = 1;
    while (nxt < starts.size() && steps[nxt] == 1 &&
           input_dimensions[nxt] == output_dims[nxt]) {
      running_size *= input_dimensions[nxt];
      ++nxt;
    }
    if (running_size > 1) {
      p_flattened_input_dims->emplace_back(running_size);
      p_flattened_output_dims->emplace_back(running_size);
      starts[cur] = 0LL;
      ends[cur] = running_size;
      steps[cur] = 1LL;
      ++cur;
    }
  }

  // No actual slice dim, and all dims are size 1.
  if (cur == 0) {
    p_flattened_input_dims->emplace_back(1LL);
    p_flattened_output_dims->emplace_back(1LL);
    starts[cur] = 0LL;
    ends[cur] = 1LL;
    steps[cur] = 1LL;
    ++cur;
  }

  if (p_flattened_output_dims->size() == output_dims.size()) {
    p_flattened_input_dims->clear();
    p_flattened_output_dims->clear();
    p_flattened_input_dims = nullptr;
    p_flattened_output_dims = nullptr;
  } else {
    starts.resize(cur);
    ends.resize(cur);
    steps.resize(cur);
  }
}

// Slice V1-9 & DynamicSlice
bool SliceBase::PrepareForCompute(
    gsl::span<const int64_t> raw_starts, gsl::span<const int64_t> raw_ends,
    gsl::span<const int64_t> raw_axes,
    SliceOp::PrepareForComputeMetadata& compute_metadata) {
  assert(SliceOp::PrepareForComputeHelper(raw_starts, raw_ends, raw_axes,
                                          compute_metadata));
  FlattenOutputDims(compute_metadata.input_dimensions_,
                    compute_metadata.output_dims_, compute_metadata.starts_,
                    compute_metadata.ends_, compute_metadata.steps_,
                    compute_metadata.p_flattened_input_dims_,
                    compute_metadata.p_flattened_output_dims_);
  return true;
}

// DynamicSlice & Slice V10
bool SliceBase::PrepareForCompute(
    gsl::span<const int64_t> raw_starts, gsl::span<const int64_t> raw_ends,
    gsl::span<const int64_t> raw_axes, gsl::span<const int64_t> raw_steps,
    SliceOp::PrepareForComputeMetadata& compute_metadata) {
  assert(SliceOp::PrepareForComputeHelper(raw_starts, raw_ends, raw_axes,
                                          raw_steps, compute_metadata));
  FlattenOutputDims(compute_metadata.input_dimensions_,
                    compute_metadata.output_dims_, compute_metadata.starts_,
                    compute_metadata.ends_, compute_metadata.steps_,
                    compute_metadata.p_flattened_input_dims_,
                    compute_metadata.p_flattened_output_dims_);

  return true;
}

}  // namespace Cudnn
