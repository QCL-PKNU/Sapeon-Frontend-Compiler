// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// #pragma once
#include <stdint.h>

#include <algorithm>
#include <iostream>

#include "cudnn/common/binary_elementwise.hpp"
#include "cudnn/common/binary_elementwise_impl.cuh"
#include "cudnn/common/common.cuh"
#include "cudnn/common/cuda_utils.hpp"
#include "gsl-lite.hpp"

namespace Cudnn {
// broadcast by computing output coordinate from offset, using fast_divmod
template <typename T, typename T1, typename T2, typename FuncT,
          bool lhs_need_compute, bool rhs_need_compute, int NumThreadsPerBlock,
          int NumElementsPerThread>
__global__ void _BinaryElementWise(
    int32_t output_rank, const TArray<int64_t> lhs_padded_strides,
    const T1* lhs_data, const TArray<int64_t> rhs_padded_strides,
    const T2* rhs_data, const TArray<fast_divmod> fdm_output_strides,
    T* output_data, const FuncT& functor, CUDA_LONG N) {
  CUDA_LONG start =
      NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  T1 lvalue[NumElementsPerThread];
  T2 rvalue[NumElementsPerThread];

  CUDA_LONG id = start;
#pragma unrollusr / local / cuda / bin
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      CUDA_LONG lhs_index = (lhs_need_compute ? 0 : id);
      CUDA_LONG rhs_index = (rhs_need_compute ? 0 : id);
      // compute indexes with broadcasting rules:
      // https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
      CUDA_LONG offset = id;
#pragma unroll
      for (auto dim = 0; dim < fdm_output_strides.Capacity(); dim++) {
        if (dim >= output_rank) {
          break;
        }
        int q, r;
        fdm_output_strides[dim].divmod(offset, q, r);
        if (lhs_need_compute) {
          lhs_index += static_cast<int>(lhs_padded_strides[dim]) * q;
        }

        if (rhs_need_compute) {
          rhs_index += static_cast<int>(rhs_padded_strides[dim]) * q;
        }
        offset = r;
      }
      lvalue[i] = lhs_data[lhs_index];
      rvalue[i] = rhs_data[rhs_index];

      id += NumThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = functor(lvalue[i], rvalue[i]);

      id += NumThreadsPerBlock;
    }
  }
}

// for scalar broadcast or non-broadcast case
template <bool IncL, bool IncR, typename T, typename T1, typename T2,
          typename FuncT, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _BinaryElementWiseSimple(const T1* lhs_data, const T2* rhs_data,
                                         T* output_data, const FuncT& func,
                                         CUDA_LONG N) {
  CUDA_LONG start =
      NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  T1 lvalue[NumElementsPerThread];
  T2 rvalue[NumElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      lvalue[i] = lhs_data[IncL ? id : 0];
      rvalue[i] = rhs_data[IncR ? id : 0];

      id += NumThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = func(lvalue[i], rvalue[i]);

      id += NumThreadsPerBlock;
    }
  }
}

// for rhs per-channel broadcast case
template <typename T, typename T1, typename T2, typename FuncT,
          int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _BinaryElementWiseRhsPerChannelBatch1(const T1* lhs_data,
                                                      const T2* rhs_data,
                                                      const fast_divmod fdm_H,
                                                      T* output_data,
                                                      FuncT func, CUDA_LONG N) {
  CUDA_LONG start =
      NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  T1 lvalue[NumElementsPerThread];
  T2 rvalue[NumElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      CUDA_LONG rhs_id = fdm_H.div(id);
      lvalue[i] = lhs_data[id];
      rvalue[i] = rhs_data[rhs_id];

      id += NumThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = func(lvalue[i], rvalue[i]);

      id += NumThreadsPerBlock;
    }
  }
}

template <typename T, typename T1, typename T2, typename FuncT,
          int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _BinaryElementWiseRhsPerChannelBatchN(
    const T1* lhs_data, const T2* rhs_data, const fast_divmod fdm_H,
    const fast_divmod fdm_C, T* output_data, FuncT func, CUDA_LONG N) {
  CUDA_LONG start =
      NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  T1 lvalue[NumElementsPerThread];
  T2 rvalue[NumElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      CUDA_LONG rhs_id = fdm_H.div(id);
      int q, r;
      fdm_C.divmod(rhs_id, q, r);
      rhs_id = r;

      lvalue[i] = lhs_data[id];
      rvalue[i] = rhs_data[rhs_id];

      id += NumThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = func(lvalue[i], rvalue[i]);

      id += NumThreadsPerBlock;
    }
  }
}

template <typename T, typename T1, typename T2, typename FuncT>
void BinaryElementWiseNoBroadcastImpl(cudaStream_t stream, const T1* lhs_data,
                                      const T2* rhs_data, T* output_data,
                                      const FuncT& func, size_t count) {
  if (count ==
      0)  // special case where there's a dim value of 0 in the output shape
    return;

#ifdef USE_ROCM
  const int num_elements_per_thread = 2;
  const int num_threads_per_block = 512;
#else
  const int num_elements_per_thread = GridDim::maxElementsPerThread;
  const int num_threads_per_block = GridDim::maxThreadsPerBlock;
#endif

  int blocksPerGrid = static_cast<int>(
      CeilDiv(count, num_threads_per_block * num_elements_per_thread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _BinaryElementWiseSimple<true, true, T, T1, T2, FuncT, num_threads_per_block,
                           num_elements_per_thread>
      <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
          lhs_data, rhs_data, output_data, func, N);
}

template <typename T, typename T1, typename T2, typename FuncT>
void BinaryElementWiseImpl(
    cudaStream_t stream, int32_t output_rank_or_simple_broadcast,
    const TArray<int64_t>* lhs_padded_strides, const T1* lhs_data,
    const TArray<int64_t>* rhs_padded_strides, const T2* rhs_data,
    const TArray<fast_divmod>* fdm_output_strides, const fast_divmod& fdm_H,
    const fast_divmod& fdm_C, T* output_data, const FuncT& func, size_t count) {
  if (count == 0) return;

  const int num_elements_per_thread = GridDim::maxElementsPerThread;
  const int num_threads_per_block = GridDim::maxThreadsPerBlock;

  int blocksPerGrid = static_cast<int>(
      CeilDiv(count, num_threads_per_block * num_elements_per_thread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  if (output_rank_or_simple_broadcast ==
      static_cast<int32_t>(SimpleBroadcast::NoBroadcast)) {
    _BinaryElementWiseSimple<true, true, T, T1, T2, FuncT,
                             num_threads_per_block, num_elements_per_thread>
        <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
            lhs_data, rhs_data, output_data, func, N);
  } else if (output_rank_or_simple_broadcast ==
             static_cast<int32_t>(SimpleBroadcast::LeftScalar)) {
    _BinaryElementWiseSimple<false, true, T, T1, T2, FuncT,
                             num_threads_per_block, num_elements_per_thread>
        <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
            lhs_data, rhs_data, output_data, func, N);
  } else if (output_rank_or_simple_broadcast ==
             static_cast<int32_t>(SimpleBroadcast::RightScalar)) {
    _BinaryElementWiseSimple<true, false, T, T1, T2, FuncT,
                             num_threads_per_block, num_elements_per_thread>
        <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
            lhs_data, rhs_data, output_data, func, N);
  } else if (output_rank_or_simple_broadcast ==
             static_cast<int32_t>(SimpleBroadcast::RightPerChannelBatch1)) {
    _BinaryElementWiseRhsPerChannelBatch1<
        T, T1, T2, FuncT, num_threads_per_block, num_elements_per_thread>
        <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
            lhs_data, rhs_data, fdm_H, output_data, func, N);
  } else if (output_rank_or_simple_broadcast ==
             static_cast<int32_t>(SimpleBroadcast::RightPerChannelBatchN)) {
    _BinaryElementWiseRhsPerChannelBatchN<
        T, T1, T2, FuncT, num_threads_per_block, num_elements_per_thread>
        <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
            lhs_data, rhs_data, fdm_H, fdm_C, output_data, func, N);
  } else {
    if (lhs_padded_strides && rhs_padded_strides &&
        lhs_padded_strides->Size() && rhs_padded_strides->Size())
      _BinaryElementWise<T, T1, T2, FuncT, true, true, num_threads_per_block,
                         num_elements_per_thread>
          <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
              output_rank_or_simple_broadcast, *lhs_padded_strides, lhs_data,
              *rhs_padded_strides, rhs_data, *fdm_output_strides, output_data,
              func, N);
    else if (lhs_padded_strides && lhs_padded_strides->Size())
      _BinaryElementWise<T, T1, T2, FuncT, true, false, num_threads_per_block,
                         num_elements_per_thread>
          <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
              output_rank_or_simple_broadcast, *lhs_padded_strides, lhs_data,
              TArray<int64_t>(),  // rhs is not computed, so no need to
                                  // deference rhs_padded_strides
              rhs_data, *fdm_output_strides, output_data, func, N);
    else if (rhs_padded_strides && rhs_padded_strides->Size())
      _BinaryElementWise<T, T1, T2, FuncT, false, true, num_threads_per_block,
                         num_elements_per_thread>
          <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
              output_rank_or_simple_broadcast,
              TArray<int64_t>(),  // lhs is not computed, so no need to
                                  // deference lhs_padded_strides
              lhs_data, *rhs_padded_strides, rhs_data, *fdm_output_strides,
              output_data, func, N);
  }
}

// 템플릿 함수는 헤더에서 구현되거나 미리 사용하는 파라미터를 입력한 구현부가
// 있어서 컴파일러가 아 이거 쓰는구나 라고 알아야 함 따라서 미리 등록이 쉽게
// 하기 위해서 매크로를 등록함 (실제 이함수는 쓰이지 않음)
// BinaryElementWiseImpl를 다양하게 쓰게 하기 위해서임 이용한 함수는
// /cuda/binary_elementwise.hpp 에 있음 예로
// Impl_Add 함수를 볼것
#define REGISTER_TEMPLATE_TYPE(type_name, func_name)                           \
  void register_template_##type_name_##func_name(                              \
      cudaStream_t stream, int32_t output_rank_or_simple_broadcast,            \
      TArray<int64_t>* lhs_padded_strides, type_name* lhs_data,                \
      TArray<int64_t>* rhs_padded_strides, type_name* rhs_data,                \
      TArray<fast_divmod>* fdm_output_strides, fast_divmod fdm_H,              \
      fast_divmod fdm_C, type_name* output_data,                               \
      OP_##func_name<type_name, type_name, type_name> act_func,                \
      size_t count) {                                                          \
    BinaryElementWiseImpl<type_name, type_name, type_name,                     \
                          OP_##func_name<type_name, type_name, type_name>>(    \
        stream, output_rank_or_simple_broadcast, lhs_padded_strides, lhs_data, \
        rhs_padded_strides, rhs_data, fdm_output_strides, fdm_H, fdm_C,        \
        output_data, act_func, count);                                         \
  }

REGISTER_TEMPLATE_TYPE(double, Add)
REGISTER_TEMPLATE_TYPE(float, Add)
REGISTER_TEMPLATE_TYPE(int32_t, Add)
REGISTER_TEMPLATE_TYPE(int16_t, Add)
REGISTER_TEMPLATE_TYPE(int8_t, Add)
REGISTER_TEMPLATE_TYPE(uint8_t, Add)
REGISTER_TEMPLATE_TYPE(half, Add)

REGISTER_TEMPLATE_TYPE(double, Mul)
REGISTER_TEMPLATE_TYPE(float, Mul)
REGISTER_TEMPLATE_TYPE(int32_t, Mul)
REGISTER_TEMPLATE_TYPE(int16_t, Mul)
REGISTER_TEMPLATE_TYPE(int8_t, Mul)
REGISTER_TEMPLATE_TYPE(uint8_t, Mul)
REGISTER_TEMPLATE_TYPE(half, Mul)

REGISTER_TEMPLATE_TYPE(double, Sub)
REGISTER_TEMPLATE_TYPE(float, Sub)
REGISTER_TEMPLATE_TYPE(int32_t, Sub)
REGISTER_TEMPLATE_TYPE(int16_t, Sub)
REGISTER_TEMPLATE_TYPE(int8_t, Sub)
REGISTER_TEMPLATE_TYPE(uint8_t, Sub)
REGISTER_TEMPLATE_TYPE(half, Sub)

REGISTER_TEMPLATE_TYPE(double, PRelu)
REGISTER_TEMPLATE_TYPE(float, PRelu)
REGISTER_TEMPLATE_TYPE(int32_t, PRelu)
REGISTER_TEMPLATE_TYPE(int16_t, PRelu)
REGISTER_TEMPLATE_TYPE(int8_t, PRelu)
REGISTER_TEMPLATE_TYPE(uint8_t, PRelu)
REGISTER_TEMPLATE_TYPE(half, PRelu)

REGISTER_TEMPLATE_TYPE(double, Min)
REGISTER_TEMPLATE_TYPE(float, Min)
REGISTER_TEMPLATE_TYPE(int32_t, Min)
REGISTER_TEMPLATE_TYPE(int16_t, Min)
REGISTER_TEMPLATE_TYPE(int8_t, Min)
REGISTER_TEMPLATE_TYPE(uint8_t, Min)
REGISTER_TEMPLATE_TYPE(half, Min)

REGISTER_TEMPLATE_TYPE(double, Max)
REGISTER_TEMPLATE_TYPE(float, Max)
REGISTER_TEMPLATE_TYPE(int32_t, Max)
REGISTER_TEMPLATE_TYPE(int16_t, Max)
REGISTER_TEMPLATE_TYPE(int8_t, Max)
REGISTER_TEMPLATE_TYPE(uint8_t, Max)
REGISTER_TEMPLATE_TYPE(half, Max)

REGISTER_TEMPLATE_TYPE(double, Div)
REGISTER_TEMPLATE_TYPE(float, Div)
REGISTER_TEMPLATE_TYPE(half, Div)
}  // namespace Cudnn
