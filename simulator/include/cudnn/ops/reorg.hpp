#ifndef CUDNN_OPS_REORG_HPP
#define CUDNN_OPS_REORG_HPP

#include <cudnn.h>

#include <memory>

#include "network/descriptor.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cudnn_operation.hpp"

namespace Cudnn {
template <typename Type, cudnnDataType_t DataType>
class Reorg : public CudnnOperation<Type> {
 public:
  static std::unique_ptr<CudnnOperation<Type>> Create();
  std::shared_ptr<Tensor> Forward(cudnnHandle_t &handle, Layer &layer) override;

 private:
  void InitOutputTensor();
  void OperationForward();
  std::shared_ptr<Tensor> input_;
  std::shared_ptr<Tensor> output_;
  std::shared_ptr<Descriptor> sampling_;
};
}  // namespace Cudnn

#endif  // CUDNN_OPS_REORG_HPP
