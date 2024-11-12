#ifndef CUDNN_COMMON_BINARY_ELEMENTWISE_HPP
#define CUDNN_COMMON_BINARY_ELEMENTWISE_HPP

#include <stdint.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "cudnn/common/common.cuh"
#include "cudnn/common/cuda_utils.hpp"
#include "gsl-lite.hpp"

namespace Cudnn {
#define BINARY_OPS()                                 \
  BINARY_OP_NAME_EXPR(Add, (a + b))                  \
  BINARY_OP_NAME_EXPR(Sub, (a - b))                  \
  BINARY_OP_NAME_EXPR(Mul, (a * b))                  \
  BINARY_OP_NAME_EXPR(Div, (a / b))                  \
  BINARY_OP_NAME_EXPR(Pow, _Pow(a, b))               \
  BINARY_OP_NAME_EXPR(And, (a & b))                  \
  BINARY_OP_NAME_EXPR(Or, (a | b))                   \
  BINARY_OP_NAME_EXPR(Xor, (a ^ b))                  \
  BINARY_OP_NAME_EXPR(PRelu, (a > (T)0 ? a : a * b)) \
  BINARY_OP_NAME_EXPR(Max, _Max(a, b))               \
  BINARY_OP_NAME_EXPR(Min, _Min(a, b))               \
  BINARY_OP_NAME_EXPR(Mod, _Mod(a, b))               \
  BINARY_OP_NAME_EXPR(Fmod, _Fmod(a, b))

#define OP_FUNCTOR_DEFINITION(name, expr)                                   \
  template <class T, class T1, class T2>                                    \
  struct OP_##name {                                                        \
    __device__ __inline__ T operator()(T1 a, T2 b) const { return (expr); } \
  };

#define BINARY_OP_NAME_EXPR(name, expr) OP_FUNCTOR_DEFINITION(name, expr)

BINARY_OPS()

int BinaryElementwiseBroadcastPrepare(const std::vector<int64_t>& lhs_shape,
                                      const std::vector<int64_t>& rhs_shape,
                                      const std::vector<int64_t>& output_shape,
                                      int32_t& output_rank_or_simple_broadcast,
                                      fast_divmod& fdm_H, fast_divmod& fdm_C,
                                      TArray<int64_t>& lhs_padded_strides,
                                      TArray<int64_t>& rhs_padded_strides,
                                      TArray<fast_divmod>& fdm_output_strides);

int ComputeOutputShape(const std::string& node_name,
                       const std::vector<int64_t>& lhs_shape,
                       const std::vector<int64_t>& rhs_shape,
                       std::vector<int64_t>& out_shape);

template <typename T, typename T1, typename T2, typename FuncT>
void BinaryElementWiseImpl(
    cudaStream_t stream, int32_t output_rank_or_simple_broadcast,
    const TArray<int64_t>* lhs_padded_strides, const T1* lhs_data,
    const TArray<int64_t>* rhs_padded_strides, const T2* rhs_data,
    const TArray<fast_divmod>* fdm_output_strides, const fast_divmod& fdm_H,
    const fast_divmod& fdm_C, T* output_data, const FuncT& func, size_t count);

#define DEFINE_IMPL_BINARY(ImplName)                                           \
  template <typename T>                                                        \
  void Impl_##ImplName(                                                        \
      cudaStream_t stream, int32_t output_rank_or_simple_broadcast,            \
      const TArray<int64_t>* lhs_padded_strides, const T* lhs_data,            \
      const TArray<int64_t>* rhs_padded_strides, const T* rhs_data,            \
      const TArray<fast_divmod>* fdm_output_strides, const fast_divmod& fdm_H, \
      const fast_divmod& fdm_C, T* output_data, size_t count) {                \
    BinaryElementWiseImpl(stream, output_rank_or_simple_broadcast,             \
                          lhs_padded_strides, lhs_data, rhs_padded_strides,    \
                          rhs_data, fdm_output_strides, fdm_H, fdm_C,          \
                          output_data, OP_##ImplName<T, T, T>(), count);       \
  }

DEFINE_IMPL_BINARY(Add)
DEFINE_IMPL_BINARY(Mul)
DEFINE_IMPL_BINARY(Sub)
DEFINE_IMPL_BINARY(PRelu)
DEFINE_IMPL_BINARY(Min)
DEFINE_IMPL_BINARY(Max)
DEFINE_IMPL_BINARY(Div)

#undef DEFINE_IMPL_BINARY

}  // namespace Cudnn

#endif  // CUDNN_COMMON_BINARY_ELEMENTWISE_HPP
