// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CUDNN_COMMON_BINARY_ELEMENTWISE_OPS_HPP
#define CUDNN_COMMON_BINARY_ELEMENTWISE_OPS_HPP

#include <algorithm>
#include <string>
#include <vector>

#include "cudnn/common/cuda_utils.hpp"
#include "cudnn/common/fast_divmod.hpp"
#include "cudnn/common/tensor_shape.hpp"
#include "cudnn/common/utils.hpp"

namespace Cudnn {

struct BinaryElementwisePreparation {
  int32_t output_rank_or_simple_broadcast =
      0;  // for no_broadcast|left_scalar|right_scalar cases, output_rank uses
          // SimpleBroadcast enums

  TArray<int64_t> lhs_padded_strides;
  TArray<int64_t> rhs_padded_strides;
  TArray<fast_divmod> fdm_output_strides;

  // these are for RightPerChannel case
  fast_divmod fdm_H;
  fast_divmod fdm_C;

  BinaryElementwisePreparation() {}

  bool BinaryElementwiseBroadcastPrepareHelper(
      const TensorShape& lhs_shape, const TensorShape& rhs_shape,
      const TensorShape& output_shape) {
    int32_t lhs_rank = gsl::narrow_cast<int32_t>(lhs_shape.NumDimensions());
    int32_t rhs_rank = gsl::narrow_cast<int32_t>(rhs_shape.NumDimensions());
    int32_t out_rank = std::max(lhs_rank, rhs_rank);

    // early return when shapes match
    if (lhs_shape == rhs_shape) {
      output_rank_or_simple_broadcast =
          static_cast<int32_t>(SimpleBroadcast::NoBroadcast);
      return true;
    }

    // early return if one operand is scalar
    if (lhs_shape.Size() == 1 || rhs_shape.Size() == 1) {
      output_rank_or_simple_broadcast = static_cast<int32_t>(
          lhs_shape.Size() == 1 ? SimpleBroadcast::LeftScalar
                                : SimpleBroadcast::RightScalar);
      return true;
    }

    // special case for lhs(N,C,H) and rhs (C,1) which is used in conv bias
    // when N == 1: out[id] = op(lhs[id], rhs[id / H])
    // When N > 1:  out[id] = op(lhs[id], rhs[id / H % C])
    if (lhs_shape == output_shape) {
      const auto& rhs_dims = rhs_shape.GetDims();
      int64_t C = 0;
      if (1 ==
          std::count_if(rhs_dims.begin(), rhs_dims.end(), [&C](int64_t dim) {
            if (dim != 1) C = dim;
            return (dim != 1);
          })) {
        int32_t dim_C = gsl::narrow_cast<int32_t>(
            std::find(rhs_dims.begin(), rhs_dims.end(), C) - rhs_dims.begin() +
            output_shape.NumDimensions() - rhs_shape.NumDimensions());
        int64_t N = output_shape.SizeToDimension(dim_C);
        int64_t H = (dim_C < out_rank - 1 ? output_shape.SizeFromDimension(
                                                static_cast<size_t>(dim_C) + 1)
                                          : 1);

        std::vector<int64_t> new_output_dims;
        if (N == 1) {
          output_rank_or_simple_broadcast =
              static_cast<int32_t>(SimpleBroadcast::RightPerChannelBatch1);
          fdm_H = fast_divmod(gsl::narrow_cast<int>(H));
        } else {
          output_rank_or_simple_broadcast =
              static_cast<int32_t>(SimpleBroadcast::RightPerChannelBatchN);
          fdm_H = fast_divmod(gsl::narrow_cast<int>(H));
          fdm_C = fast_divmod(gsl::narrow_cast<int>(C));
        }
        return true;
      }
    }

    output_rank_or_simple_broadcast = out_rank;

    if (lhs_shape != output_shape) {
      TensorPitches original_lhs_padded_strides(lhs_shape.GetDims(), out_rank);
      lhs_padded_strides.SetSize(out_rank);
      auto offset = out_rank - lhs_rank;
      for (auto i = offset; i < out_rank; ++i) {
        // the stride for broadcast dimension is kept as 0
        if (lhs_shape.GetDims()[static_cast<size_t>(i) - offset] != 1) {
          lhs_padded_strides[i] = original_lhs_padded_strides[i];
        }
      }
    }

    if (rhs_shape != output_shape) {
      TensorPitches original_rhs_padded_strides(rhs_shape.GetDims(), out_rank);
      rhs_padded_strides.SetSize(out_rank);
      auto offset = out_rank - rhs_rank;
      for (auto i = offset; i < out_rank; ++i) {
        // the stride for broadcast dimension is kept as 0
        if (rhs_shape.GetDims()[static_cast<size_t>(i) - offset] != 1) {
          rhs_padded_strides[i] = original_rhs_padded_strides[i];
        }
      }
    }

    TensorPitches original_output_strides(output_shape.GetDims());
    fdm_output_strides.SetSize(out_rank);
    for (auto i = 0; i < out_rank; ++i) {
      fdm_output_strides[i] =
          fast_divmod(gsl::narrow_cast<int>(original_output_strides[i]));
    }

    return true;
  }
};

bool ComputeOutputShape(const std::string& node_name,
                        const TensorShape& lhs_shape,
                        const TensorShape& rhs_shape, TensorShape& out_shape);

}  // namespace Cudnn

#endif  // CUDNN_COMMON_BINARY_ELEMENTWISE_OPS_HPP
