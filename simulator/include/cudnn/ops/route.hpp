#ifndef CUDNN_OPS_ROUTE_HPP
#define CUDNN_OPS_ROUTE_HPP

#include <cudnn.h>

#include <memory>
#include <vector>

#include "network/descriptor.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cudnn_operation.hpp"

namespace Cudnn {
template <typename Type, cudnnDataType_t DataType>
class Route : public CudnnOperation<Type> {
 public:
  static std::unique_ptr<CudnnOperation<Type>> Create();
  std::shared_ptr<Tensor> Forward(cudnnHandle_t &handle, Layer &layer) override;

 private:
  void InitOutputTensor();
  void OperationForward();
  size_t num_inputs_;
  size_t num_channels_;
  std::vector<std::shared_ptr<Tensor>> inputs_;
  std::shared_ptr<Tensor> output_;
};
}  // namespace Cudnn

#endif  // CUDNN_OPS_ROUTE_HPP
