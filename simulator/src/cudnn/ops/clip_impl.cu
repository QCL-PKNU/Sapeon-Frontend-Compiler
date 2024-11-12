// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cudnn/common/common.cuh"
#include "cudnn/common/cuda_common.hpp"
#include "cudnn/ops/clip_impl.h"

namespace Cudnn {
template <typename T>
__global__ void _Clip(const T* input, T* output, T min_default, T max_default,
                      size_t N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  output[id] = (input[id] < min_default)
                   ? min_default
                   : ((input[id] > max_default) ? max_default : input[id]);
}

template <typename T>
void ClipImpl(cudaStream_t stream, const T* input_data, T* output_data,
              T min_default, T max_default, size_t count) {
  typedef typename ToCudaType<T>::MappedType CudaT;

  int blocksPerGrid =
      (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  union ConstAliasUnion {
    const T* t;
    const CudaT* cudaT;
    ConstAliasUnion(const T* _t) { t = _t; }
  };
  union AliasUnion {
    T* t;
    CudaT* cudaT;
    AliasUnion(T* _t) { t = _t; }
  };
  _Clip<CudaT><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      ((union ConstAliasUnion)input_data).cudaT,
      ((union AliasUnion)output_data).cudaT,
      *((union AliasUnion) & min_default).cudaT,
      *((union AliasUnion) & max_default).cudaT, count);
}

template void ClipImpl<float>(cudaStream_t stream, const float* input_data,
                              float* output_data, float min_default,
                              float max_default, size_t count);
template void ClipImpl<double>(cudaStream_t stream, const double* input_data,
                               double* output_data, double min_default,
                               double max_default, size_t count);
/*
template void ClipImpl<MLFloat16>(cudaStream_t stream,
                                  const MLFloat16* input_data,
                                  MLFloat16* output_data, const MLFloat16* min,
                                  const MLFloat16* max, MLFloat16 min_default,
                                  MLFloat16 max_default, size_t count);
*/
template void ClipImpl<int8_t>(cudaStream_t stream, const int8_t* input_data,
                               int8_t* output_data, int8_t min_default,
                               int8_t max_default, size_t count);
template void ClipImpl<uint8_t>(cudaStream_t stream, const uint8_t* input_data,
                                uint8_t* output_data, uint8_t min_default,
                                uint8_t max_default, size_t count);
template void ClipImpl<int64_t>(cudaStream_t stream, const int64_t* input_data,
                                int64_t* output_data, int64_t min_default,
                                int64_t max_default, size_t count);
template void ClipImpl<uint64_t>(cudaStream_t stream,
                                 const uint64_t* input_data,
                                 uint64_t* output_data, uint64_t min_default,
                                 uint64_t max_default, size_t count);

}  // namespace Cudnn
