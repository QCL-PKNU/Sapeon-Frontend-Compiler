#ifndef CUDNN_OPS_BATCH_NORM_HELPER_HPP
#define CUDNN_OPS_BATCH_NORM_HELPER_HPP

#include <cassert>
#include <sstream>
#include <vector>

#include "cudnn/common/tensor_shape.hpp"
namespace Cudnn {
class BatchNormHelper {
 public:
  static void NormalizeDims(const TensorShape& x_shape,
                            std::vector<int64_t>& new_dims) {
    new_dims.clear();
    auto orig_dims = x_shape.GetDims();
    assert(orig_dims.size() < 6);
    if (orig_dims.size() == 4 /*supported size by CUDA*/ ||
        orig_dims.size() == 5 /*supported size by CUDA*/) {
      new_dims = std::vector<int64_t>(orig_dims.begin(), orig_dims.end());
      return;
    }

    auto rank = x_shape.NumDimensions();
    auto num_samples = rank > 0 ? orig_dims[0] : 1;  // NCHW
    auto num_channels = rank > 1 ? orig_dims[1] : 1;
    auto height = rank > 2 ? orig_dims[2] : 1;
    int64_t width = 1;
    new_dims = {num_samples, num_channels, height, width};
  }
};
}  // namespace Cudnn

#endif  // CUDNN_OPS_BATCH_NORM_HELPER_HPP
