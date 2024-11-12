#ifndef CUDNN_OPS_CONV_TRANSPOSE_HPP
#define CUDNN_OPS_CONV_TRANSPOSE_HPP

#include <cudnn.h>

#include <memory>
#include <vector>

#include "cudnn/common/conv_transpose_attributes.hpp"
#include "cudnn/ops/conv_base.hpp"
#include "network/descriptor.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cudnn_operation.hpp"

namespace Cudnn {
template <typename Type, cudnnDataType_t DataType>
class ConvTranspose : public CudnnOperation<Type> {
 public:
  friend class OpTest;

  static std::unique_ptr<CudnnOperation<Type>> Create();
  std::shared_ptr<Tensor> Forward(cudnnHandle_t &handle, Layer &layer) override;

 private:
  void AllocateMemory();
  void OperationForward();
  void GetOutput();
  void DeAllocateMemory();
  void SetOptions(Layer &layer);
  std::vector<std::shared_ptr<Tensor>> inputs_;
  std::shared_ptr<Tensor> output_;
  cudnnHandle_t handle_;
  void *data_input_[4];
  Type *data_output_;

  size_t inputs_count_;

  ConvTransposeAttributes conv_transpose_attrs_;
  mutable CudnnConvState<cudnnConvolutionBwdDataAlgoPerf_t> s_;
};

}  // namespace Cudnn

#endif  // CUDNN_OPS_CONV_TRANSPOSE_HPP
