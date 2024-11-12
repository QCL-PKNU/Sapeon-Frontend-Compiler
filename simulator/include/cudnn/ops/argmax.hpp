#ifndef CUDNN_OPS_ARGMAX_HPP
#define CUDNN_OPS_ARGMAX_HPP

#include <cudnn.h>

#include <memory>
#include <vector>

#include "cudnn/common/tensor_shape.hpp"
#include "network/descriptor.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cudnn_operation.hpp"

namespace Cudnn {
template <typename Type, cudnnDataType_t DataType>
class ArgMax : public CudnnOperation<Type> {
 public:
  friend class OpTest;

  static std::unique_ptr<CudnnOperation<Type>> Create();
  std::shared_ptr<Tensor> Forward(cudnnHandle_t &handle, Layer &layer) override;

 private:
  void AllocateMemory();
  void GetOutput();
  void DeAllocateMemory();
  void SetOptions(Layer &layer);
  std::vector<std::shared_ptr<Tensor>> inputs_;
  std::shared_ptr<Tensor> output_;
  cudnnHandle_t handle_;
  Type **data_input_;
  int64_t *data_output_;
  int inputs_count_;

  TensorShapeVector axes_;
  bool keepdims_;
  bool noop_with_empty_axes_;
  bool select_last_index_;
};
}  // namespace Cudnn

#endif  // CUDNN_OPS_ARGMAX_HPP
