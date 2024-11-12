// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// #pragma once
#include "cudnn/common/binary_elementwise.hpp"

#include <stdint.h>

#include <algorithm>
#include <iostream>

#include "cudnn/common/common.cuh"
#include "cudnn/common/cuda_utils.hpp"
#include "cudnn/common/utils.hpp"
#include "gsl-lite.hpp"

namespace Cudnn {
int BinaryElementwiseBroadcastPrepare(const std::vector<int64_t>& lhs_shape,
                                      const std::vector<int64_t>& rhs_shape,
                                      const std::vector<int64_t>& output_shape,
                                      int32_t& output_rank_or_simple_broadcast,
                                      fast_divmod& fdm_H, fast_divmod& fdm_C,
                                      TArray<int64_t>& lhs_padded_strides,
                                      TArray<int64_t>& rhs_padded_strides,
                                      TArray<fast_divmod>& fdm_output_strides) {
  int32_t lhs_rank = lhs_shape.size();
  int32_t rhs_rank = rhs_shape.size();
  int32_t out_rank = std::max(lhs_rank, rhs_rank);

  // early return when shapes match
  if (lhs_shape == rhs_shape) {
    output_rank_or_simple_broadcast =
        static_cast<int32_t>(SimpleBroadcast::NoBroadcast);
    return 1;
  }

  // early return if one operand is scalar
  if (lhs_shape.size() == 1 || rhs_shape.size() == 1) {
    output_rank_or_simple_broadcast = static_cast<int32_t>(
        lhs_shape.size() == 1 ? SimpleBroadcast::LeftScalar
                              : SimpleBroadcast::RightScalar);
    return 1;
  }

  // special case for lhs(N,C,H) and rhs (C,1) which is used in conv bias
  // when N == 1: out[id] = op(lhs[id], rhs[id / H])
  // When N > 1:  out[id] = op(lhs[id], rhs[id / H % C])
  if (lhs_shape == output_shape) {
    const auto& rhs_dims = rhs_shape;
    int64_t C = 0;

    if (1 == std::count_if(rhs_dims.begin(), rhs_dims.end(), [&C](int64_t dim) {
          if (dim != 1) C = dim;
          return (dim != 1);
        })) {
      int32_t dim_C = std::find(rhs_dims.begin(), rhs_dims.end(), C) -
                      rhs_dims.begin() + output_shape.size() - rhs_shape.size();
      int64_t N = 1;

      if (dim_C <= output_shape.size()) {
        for (int64_t index = 0; index < dim_C; index++) {
          if (output_shape[index] < 0) {
            N = -1;
            break;
          }
          N *= output_shape[index];
        }
      }

      int64_t H = 1;

      if (dim_C < out_rank - 1) {
        for (int64_t index = static_cast<size_t>(dim_C) + 1;
             index < output_shape.size(); index++) {
          if (output_shape[index] < 0) {
            H = -1;
            break;
          }
          H *= output_shape[index];
        }
      }

      std::vector<int64_t> new_output_dims;
      if (N == 1) {
        output_rank_or_simple_broadcast =
            static_cast<int32_t>(SimpleBroadcast::RightPerChannelBatch1);
        fdm_H = fast_divmod((int)H);
      } else {
        output_rank_or_simple_broadcast =
            static_cast<int32_t>(SimpleBroadcast::RightPerChannelBatchN);
        fdm_H = fast_divmod((int)H);
        fdm_C = fast_divmod((int)C);
      }
      return 1;
    }
  }

  output_rank_or_simple_broadcast = out_rank;

  if (lhs_shape != output_shape) {
    TensorPitches original_lhs_padded_strides(lhs_shape, out_rank);
    lhs_padded_strides.SetSize(out_rank);
    auto offset = out_rank - lhs_rank;
    for (auto i = offset; i < out_rank; ++i) {
      // the stride for broadcast dimension is kept as 0
      if (lhs_shape[static_cast<size_t>(i) - offset] != 1) {
        lhs_padded_strides[i] = original_lhs_padded_strides[i];
      }
    }
  }

  if (rhs_shape != output_shape) {
    TensorPitches original_rhs_padded_strides(rhs_shape, out_rank);
    rhs_padded_strides.SetSize(out_rank);
    auto offset = out_rank - rhs_rank;
    for (auto i = offset; i < out_rank; ++i) {
      // the stride for broadcast dimension is kept as 0
      if (rhs_shape[static_cast<size_t>(i) - offset] != 1) {
        rhs_padded_strides[i] = original_rhs_padded_strides[i];
      }
    }
  }

  TensorPitches original_output_strides(output_shape);
  fdm_output_strides.SetSize(out_rank);
  for (auto i = 0; i < out_rank; ++i) {
    fdm_output_strides[i] =
        fast_divmod(gsl::narrow_cast<int>(original_output_strides[i]));
  }

  return 1;
}

int ComputeOutputShape(const std::string& node_name,
                       const std::vector<int64_t>& lhs_shape,
                       const std::vector<int64_t>& rhs_shape,
                       std::vector<int64_t>& out_shape) {
  size_t lhs_rank = lhs_shape.size();
  size_t rhs_rank = rhs_shape.size();
  size_t out_rank = std::max(lhs_rank, rhs_rank);

  std::vector<int64_t> output_dims(out_rank, 0);
  for (size_t i = 0; i < out_rank; ++i) {
    int64_t lhs_dim = 1;
    if (i < lhs_rank) lhs_dim = lhs_shape[lhs_rank - 1 - i];
    int64_t rhs_dim = 1;
    if (i < rhs_rank) rhs_dim = rhs_shape[rhs_rank - 1 - i];
    int64_t max = std::max(lhs_dim, rhs_dim);
    int64_t min = std::min(lhs_dim, rhs_dim);
    int64_t out_dim = (min == 0 ? min : max);  // special case a dim value of 0.
    if (lhs_dim != out_dim && lhs_dim != 1) {
      std::cerr << node_name << ": left operand cannot broadcast" << std::endl;
      return -1;
    }

    if (rhs_dim != out_dim && rhs_dim != 1) {
      std::cerr << node_name << ": right operand cannot broadcast" << std::endl;
      return -2;
    }

    output_dims[out_rank - 1 - i] = out_dim;
  }
  out_shape = output_dims;
  return true;
}

}  // namespace Cudnn
