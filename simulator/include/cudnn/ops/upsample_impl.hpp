// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CUDNN_OPS_UPSAMPLE_IMPL_HPP
#define CUDNN_OPS_UPSAMPLE_IMPL_HPP

#include <stdint.h>

#include "cudnn/common/cuda_utils.hpp"
#include "cudnn/ops/upsamplebase.hpp"

namespace Cudnn {

template <typename T>
void UpampleImpl(cudaStream_t stream, const UpsampleMode upsample_mode,
                 const size_t rank, const int64_t input_dim2,
                 const TArray<int64_t>& input_pitches,
                 const TArray<fast_divmod>& output_div_pitches,
                 const TArray<fast_divmod>& scales_div, const T* input_data,
                 T* output_data, const size_t N);

}  // namespace Cudnn

#endif  // CUDNN_OPS_UPSAMPLE_IMPL_HPP
