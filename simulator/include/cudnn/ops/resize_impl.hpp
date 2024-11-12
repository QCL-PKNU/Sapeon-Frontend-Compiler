// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CUDNN_OPS_RESIZE_IMPL_HPP
#define CUDNN_OPS_RESIZE_IMPL_HPP

#include <stdint.h>

#include "cudnn/common/cuda_common.hpp"
#include "cudnn/common/cuda_utils.hpp"
#include "cudnn/ops/upsamplebase.hpp"

namespace Cudnn {

size_t CalcResizeBufferSize(const UpsampleMode upsample_mode,
                            const gsl::span<const int64_t>& output_dims);

template <typename T>
void ResizeImpl(cudaStream_t stream, const UpsampleMode upsample_mode,
                const int rank, TArray<int64_t>& input_shape,
                TArray<int64_t>& output_shape, TArray<int64_t>& input_strides,
                TArray<fast_divmod>& output_div_pitches,
                TArray<float>& scales_vals, TArray<float, 10>& roi,
                const T* input_data, T* output_data, const size_t N,
                bool extrapolation_enabled, const T extrapolation_value,
                float cubic_coeff_a, bool exclude_outside,
                ResizeCoordinateTransformationMode coordinate_transform_mode,
                ResizeNearestMode nearest_mode, void* dims_mapping);

}  // namespace Cudnn

#endif  // CUDNN_OPS_RESIZE_IMPL_HPP
