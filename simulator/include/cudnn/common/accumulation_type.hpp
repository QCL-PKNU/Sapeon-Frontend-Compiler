// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CUDNN_COMMON_ACCUMULATION_TYPE_HPP
#define CUDNN_COMMON_ACCUMULATION_TYPE_HPP

#include <cuda_fp16.h>

namespace Cudnn {

// specifies the auxiliary type to use for accumulation of the given type
template <typename T>
struct AccumulationType;
template <>
struct AccumulationType<half> {
  using type = float;
};
template <>
struct AccumulationType<float> {
  using type = float;
};
template <>
struct AccumulationType<double> {
  using type = double;
};

template <typename T>
using AccumulationType_t = typename AccumulationType<T>::type;

}  // namespace Cudnn

#endif  // CUDNN_COMMON_ACCUMULATION_TYPE_HPP
