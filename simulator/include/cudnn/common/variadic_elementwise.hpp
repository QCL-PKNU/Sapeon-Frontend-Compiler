// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CUDNN_COMMON_VARIADIC_ELEMENTWISE_HPP
#define CUDNN_COMMON_VARIADIC_ELEMENTWISE_HPP

#include <cstdint>

#include "cudnn/common/cuda_utils.hpp"

namespace Cudnn {

namespace variadic_elementwise_ops {
struct Sum {};
struct Min {};
struct Max {};
}  // namespace variadic_elementwise_ops

template <typename T, typename VariadicElementwiseOpTag>
void Impl_General(cudaStream_t stream, int32_t output_rank_or_simple_broadcast,
                  const TArray<int64_t>* lhs_padded_strides, const T* lhs_data,
                  const TArray<int64_t>* rhs_padded_strides, const T* rhs_data,
                  const TArray<fast_divmod>* fdm_output_strides,
                  const fast_divmod& fdm_H, const fast_divmod& fdm_C,
                  T* output_data, size_t count);

constexpr int32_t k_max_input_batch_size = 8;

template <typename T>
using InputBatchArray = TArray<const T*, k_max_input_batch_size>;

template <typename T, typename VariadicElementwiseOpTag>
void Impl_NoBroadcastInputBatch(cudaStream_t stream,
                                InputBatchArray<T> input_data_batch,
                                T* output_data, size_t count);

}  // namespace Cudnn

#endif  // CUDNN_COMMON_VARIADIC_ELEMENTWISE_HPP
