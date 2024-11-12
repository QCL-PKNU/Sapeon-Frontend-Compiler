#ifndef CUDNN_OPS_CONVOLUTION_ONNX_HPP
#define CUDNN_OPS_CONVOLUTION_ONNX_HPP

#include <cudnn.h>

#include <memory>
#include <vector>

#include "cudnn/common/conv_attributes.hpp"
#include "cudnn/ops/conv_base.hpp"
#include "network/descriptor.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cudnn_operation.hpp"

namespace Cudnn {
template <typename Type, cudnnDataType_t DataType>
class Convolution_Onnx : public CudnnOperation<Type> {
 public:
  static std::unique_ptr<CudnnOperation<Type>> Create();
  std::shared_ptr<Tensor> Forward(cudnnHandle_t &handle, Layer &layer) override;

 private:
  void InitOutputTensor();
  void CreateDescriptors();
  void SetDescriptors();
  void SetWorkspaceSize();
  void AllocateMemory();
  void OperationForward();
  void GetOutput();
  void DeAllocateMemory();
  void DestroyDescriptors();
  std::shared_ptr<Tensor> input_;
  std::shared_ptr<Tensor> filter_;
  std::shared_ptr<Tensor> output_;
  std::shared_ptr<Tensor> bias_;
  std::shared_ptr<Descriptor> convolution_;
  cudnnHandle_t handle_;
  cudnnTensorDescriptor_t src_descriptor_;
  cudnnTensorDescriptor_t dst_descriptor_;
  cudnnFilterDescriptor_t ft_descriptor_;
  cudnnConvolutionDescriptor_t cv_descriptor_;
  cudnnConvolutionFwdAlgo_t fwd_algo_mode_;
  size_t workspace_size_;
  Type *data_input_;
  Type *data_filter_;
  Type *data_bias_;
  Type *data_workspace_;
  Type *data_output_;

  ConvAttributes conv_attrs_;
  mutable CudnnConvState<cudnnConvolutionFwdAlgoPerf_t> s_;
  constexpr static auto kDefaultConvAlgo =
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  static const cudnnConvolutionFwdAlgo_t kAllAlgos[];
};
}  // namespace Cudnn

#endif  // CUDNN_OPS_CONVOLUTION_ONNX_HPP
