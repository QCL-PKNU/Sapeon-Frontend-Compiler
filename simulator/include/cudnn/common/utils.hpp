// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CUDNN_COMMON_UTILS_HPP
#define CUDNN_COMMON_UTILS_HPP

#include <algorithm>

#include "cudnn/common/tensor_shape.hpp"
#include "gsl-lite.hpp"

namespace Cudnn {

struct TensorPitches : TensorShapeVector {
  TensorPitches(const TensorShape& shape, size_t rank = 0)
      : TensorPitches(shape.GetDims(), rank) {}
  TensorPitches(const TensorShapeVector& dims, size_t rank = 0)
      : TensorShapeVector(std::max(rank, dims.size()), 0) {
    Calculate(gsl::span<int64_t>(data(), size()), gsl::make_span(dims));
  }
  TensorPitches(const gsl::span<const int64_t>& dims, size_t rank = 0)
      : TensorShapeVector(std::max(rank, dims.size()), 0) {
    Calculate(gsl::span<int64_t>(data(), size()), dims);
  }

  static bool Calculate(const gsl::span<int64_t>& p,
                        const gsl::span<const int64_t>& dims) {
    // The pitches is the size of the next inner axis. Aka the amount to move by
    // one of the next inner axis. For a tensor with shape(2,3,4,5) the values
    // would be: (3*4*5, 4*5, 5, 1) Note that the outermost '2' is never used,
    // as you never need to move by the entire size of the outermost axis

    auto tensor_rank = dims.size();
    auto pitch_rank = p.size();
    auto padded_rank = pitch_rank - tensor_rank;
    if (static_cast<ptrdiff_t>(padded_rank) < 0) return false;

    // Guard against Scalars
    if (pitch_rank == 0) {
      return true;
    }

    *(p.rbegin()) = 1;  // The innermost axis is 1 (single values)
    if (tensor_rank > 1) {
      for (size_t i = tensor_rank - 1; i-- > 0;) {
        p.operator[](i + padded_rank) =
            p.operator[](i + 1 + padded_rank) * dims[i + 1];
      }
    }

    if (padded_rank >= 1) {
      for (size_t i = 0; i < padded_rank; ++i) {
        if (i == 0 &&
            tensor_rank >
                0)  // For scalar tensor, the values in the pitches are all 1.
          p.operator[](padded_rank - 1) = p.operator[](padded_rank) * dims[0];
        else
          p.operator[](padded_rank - 1 - i) = p.operator[](padded_rank - 1);
      }
    }
    return true;
  }
};

}  // namespace Cudnn

#endif  // CUDNN_COMMON_UTILS_HPP
