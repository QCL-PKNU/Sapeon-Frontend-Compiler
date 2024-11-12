// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cudnn/common/common.cuh"
#include "cudnn/common/cuda_common.hpp"
#include "cudnn/common/slice_impl.cuh"

namespace Cudnn {

namespace {
constexpr int kNumElementsPerThread = GridDim::maxElementsPerThread;
constexpr int kNumThreadsPerBlock = GridDim::maxThreadsPerBlock;
}  // namespace

template <bool is_grad, int DIMS, typename T>
__global__ void _SliceKernel(const TArray<int64_t> starts,
                             const TArray<int64_t> steps,
                             const TArray<int64_t> input_strides,
                             const TArray<fast_divmod> output_strides,
                             const T* input_data, T* output_data,
                             const CUDA_LONG N) {
  CUDA_LONG start =
      kNumElementsPerThread * kNumThreadsPerBlock * blockIdx.x + threadIdx.x;
  T values[kNumElementsPerThread];
  CUDA_LONG id;
  if (is_grad) {
    id = start;
#pragma unroll
    for (int i = 0; i < kNumElementsPerThread; ++i) {
      if (id < N) {
        values[i] = input_data[id];
        id += kNumThreadsPerBlock;
      }
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < kNumElementsPerThread; ++i) {
    if (id < N) {
      CUDA_LONG input_index = 0;
      int div;
      int mod = id;
      int dim = 0;
#pragma unroll
      for (; dim < DIMS - 1; ++dim) {
        output_strides[dim].divmod(mod, div, mod);
        input_index += (starts[dim] + div * steps[dim]) * input_strides[dim];
      }
      input_index += starts[dim] + mod * steps[dim];
      if (is_grad) {
        output_data[input_index] = values[i];
      } else {
        values[i] = input_data[input_index];
      }
      id += kNumThreadsPerBlock;
    }
  }

  if (!is_grad) {
    id = start;
#pragma unroll
    for (int i = 0; i < kNumElementsPerThread; ++i) {
      if (id < N) {
        output_data[id] = values[i];
        id += kNumThreadsPerBlock;
      }
    }
  }
}

template <bool is_grad>
bool SliceImplEx(cudaStream_t stream, const size_t element_size,
                 const int32_t dimension_count, const TArray<int64_t>& starts,
                 const TArray<int64_t>& steps,
                 const TArray<int64_t>& input_strides,
                 const TArray<fast_divmod>& output_strides,
                 const void* input_data, void* output_data, const size_t N) {
  int blocksPerGrid =
      static_cast<int>(CeilDiv(N, kNumThreadsPerBlock * kNumElementsPerThread));
  switch (element_size) {
#define HANDLE_DIMS(ELEMENT_TYPE, DIMS)                                    \
  case DIMS: {                                                             \
    _SliceKernel<is_grad, DIMS, ELEMENT_TYPE>                              \
        <<<blocksPerGrid, kNumThreadsPerBlock, 0, stream>>>(               \
            starts, steps, input_strides, output_strides,                  \
            reinterpret_cast<const ToCudaType<ELEMENT_TYPE>::MappedType*>( \
                input_data),                                               \
            reinterpret_cast<ToCudaType<ELEMENT_TYPE>::MappedType*>(       \
                output_data),                                              \
            (CUDA_LONG)N);                                                 \
  } break
#define HANDLE_ELEMENT_TYPE(ELEMENT_TYPE) \
  case sizeof(ELEMENT_TYPE): {            \
    switch (dimension_count) {            \
      HANDLE_DIMS(ELEMENT_TYPE, 1);       \
      HANDLE_DIMS(ELEMENT_TYPE, 2);       \
      HANDLE_DIMS(ELEMENT_TYPE, 3);       \
      HANDLE_DIMS(ELEMENT_TYPE, 4);       \
      HANDLE_DIMS(ELEMENT_TYPE, 5);       \
      HANDLE_DIMS(ELEMENT_TYPE, 6);       \
      HANDLE_DIMS(ELEMENT_TYPE, 7);       \
      HANDLE_DIMS(ELEMENT_TYPE, 8);       \
    }                                     \
  } break
    HANDLE_ELEMENT_TYPE(int8_t);
    HANDLE_ELEMENT_TYPE(int16_t);
    HANDLE_ELEMENT_TYPE(int32_t);
    HANDLE_ELEMENT_TYPE(int64_t);
    default:
      return false;
#undef HANDLE_ELEMENT_TYPE
#undef HANDLE_DIMS
  }

  return true;
}

bool SliceImpl(cudaStream_t stream, const size_t element_size,
               const int32_t dimension_count, const TArray<int64_t>& starts,
               const TArray<int64_t>& steps,
               const TArray<int64_t>& input_strides,
               const TArray<fast_divmod>& output_strides,
               const void* input_data, void* output_data, const size_t N) {
  return SliceImplEx<false>(stream, element_size, dimension_count, starts,
                            steps, input_strides, output_strides, input_data,
                            output_data, N);
}

}  // namespace Cudnn
