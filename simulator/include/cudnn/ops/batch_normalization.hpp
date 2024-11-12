#ifndef CUDNN_OPS_BATCH_NORMALIZATION_HPP
#define CUDNN_OPS_BATCH_NORMALIZATION_HPP

#include <cudnn.h>

#include <memory>

#include "network/descriptor.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cudnn_operation.hpp"

namespace Cudnn {
template <typename Type, cudnnDataType_t DataType>
class BatchNormalization : public CudnnOperation<Type> {
 public:
  static std::unique_ptr<CudnnOperation<Type>> Create();
  std::shared_ptr<Tensor> Forward(cudnnHandle_t &handle, Layer &layer) override;

 private:
  void InitOutputTensor();
  void CreateDescriptors();
  void SetDescriptors();
  void AllocateMemory();
  void OperationForward();
  void GetOutput();
  void DeAllocateMemory();
  void DestroyDescriptors();
  std::shared_ptr<Tensor> input_;
  std::shared_ptr<Tensor> mean_;
  std::shared_ptr<Tensor> scale_;
  std::shared_ptr<Tensor> variance_;
  std::shared_ptr<Tensor> output_;
  double epsilon_;
  cudnnHandle_t handle_;
  cudnnTensorDescriptor_t src_descriptor_;
  cudnnTensorDescriptor_t dst_descriptor_;
  cudnnTensorDescriptor_t bn_descriptor_;
  cudnnBatchNormMode_t bn_mode_;
  Type *data_input_;
  Type *data_bias_;
  Type *data_mean_;
  Type *data_scale_;
  Type *data_variance_;
  Type *data_output_;
};
}  // namespace Cudnn

#endif  // CUDNN_OPS_BATCH_NORMALIZATION_HPP
