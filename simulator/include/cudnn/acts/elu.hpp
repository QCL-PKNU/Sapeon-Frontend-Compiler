#ifndef CUDNN_ACTS_ELU_HPP
#define CUDNN_ACTS_ELU_HPP

#include <cudnn.h>

#include <memory>
#include <vector>

#include "cudnn/acts/activations_impl.cuh"
#include "network/descriptor.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cudnn_operation.hpp"

namespace Cudnn {
template <typename Type, cudnnDataType_t DataType>
class Elu : public CudnnOperation<Type> {
 public:
  friend class OpTest;

  static std::unique_ptr<CudnnOperation<Type>> Create();
  std::shared_ptr<Tensor> Forward(cudnnHandle_t &handle, Layer &layer) override;

 private:
  void InitOutputTensor();
  void AllocateMemory();
  void OperationForward();
  void GetOutput();
  void DeAllocateMemory();
  std::vector<std::shared_ptr<Tensor>> inputs_;
  std::shared_ptr<Tensor> output_;
  cudnnHandle_t handle_;
  Type *data_input_[1];
  Type *data_output_;

  CtxElu ctx_;
};
}  // namespace Cudnn

#endif  // CUDNN_ACTS_ELU_HPP
