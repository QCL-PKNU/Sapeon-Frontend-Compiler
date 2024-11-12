#ifndef CUDNN_OPS_GROUP_CONVOLUTION_HPP
#define CUDNN_OPS_GROUP_CONVOLUTION_HPP

#include <cudnn.h>

#include <memory>

#include "cudnn/ops/convolution.hpp"
#include "network/descriptor.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cudnn_operation.hpp"

namespace Cudnn {
template <typename Type, cudnnDataType_t DataType>
class GroupConvolution : public Convolution<Type, DataType> {
 public:
  static std::unique_ptr<CudnnOperation<Type>> Create();
};
}  // namespace Cudnn

#endif  // CUDNN_OPS_GROUP_CONVOLUTION_HPP
