#ifndef CUDNN_OPS_INSTANCE_NORM_HELPER_HPP
#define CUDNN_OPS_INSTANCE_NORM_HELPER_HPP

#include <sstream>

#include "cudnn/common/tensor_shape.hpp"
#include "glog/logging.h"

namespace Cudnn {

class InstanceNormHelper {
 public:
  static bool ValidateInputs(const TensorShape* input, const TensorShape* scale,
                             const TensorShape* B) {
    std::ostringstream ostr;

    if (input->NumDimensions() < 3) {
      std::ostringstream ostr;
      ostr << "Invalid input data: number of dimensions is less than 3: "
           << input->NumDimensions();
      return false;
    }

    if (scale->NumDimensions() != 1) {
      std::ostringstream ostr;
      LOG(ERROR) << "Invalid input scale: number of dimensions is not 1: ";
      return false;
    }

    if (scale->Size() != input->GetDims()[1]) {
      LOG(ERROR)
          << "Mismatch between input data and scale: size of scale != input "
             "channel count "
          << scale->Size() << " vs. " << input->GetDims()[1];
      return false;
    }

    if (B->NumDimensions() != 1) {
      LOG(ERROR) << "Invalid input B: number of dimensions is not 1: "
                 << B->NumDimensions();
      return false;
    }

    if (B->Size() != input->GetDims()[1]) {
      LOG(ERROR)
          << "Mismatch between input data and B: size of B != input channel "
             "count "
          << B->Size() << " vs. " << input->GetDims()[1];
      return false;
    }

    return true;
  }
};

}  // namespace Cudnn

#endif  // CUDNN_OPS_INSTANCE_NORM_HELPER_HPP
