// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CUDNN_OPS_CLIP_IMPL_H
#define CUDNN_OPS_CLIP_IMPL_H

#include <cudnn.h>

#include "cudnn/common/common.cuh"

namespace Cudnn {
template <typename T>
void ClipImpl(cudaStream_t stream, const T* input_data, T* output_data,
              T min_default, T max_default, size_t count);

}

#endif  // CUDNN_OPS_CLIP_IMPL_H
