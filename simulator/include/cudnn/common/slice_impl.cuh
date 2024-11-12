// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>

#include "cudnn/common/cuda_utils.hpp"

namespace Cudnn {

bool SliceImpl(cudaStream_t stream, const size_t element_size,
               const int32_t dimension_count, const TArray<int64_t>& starts,
               const TArray<int64_t>& steps,
               const TArray<int64_t>& input_strides,
               const TArray<fast_divmod>& output_strides,
               const void* input_data, void* output_data, const size_t N);

}  // namespace Cudnn
