#ifndef CPU_OPS_CONVOLUTION_HPP
#define CPU_OPS_CONVOLUTION_HPP

#include <memory>

#include "datatype.hpp"
#include "inference_context.hpp"
#include "network/descriptor.hpp"
#include "network/dimension.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cpu_operation.hpp"
#include "x220/quant_config.hpp"

namespace cpu {
class Convolution : public CpuOperation {
 public:
  static std::unique_ptr<CpuOperation> Create();
  void Forward(Layer &layer, InferenceContext &ctx) override;
  bool CheckValidOperation(Layer &layer, Dimension input_dimension) override;
  Dimension CalculateOutputDimension(Layer &layer,
                                     Dimension input_dimension) override;

 private:
  void InitOutputTensor(dty::DataType dtype);
  void AllocateMemory(dty::DataType dtype);
  void OperationForward(x220::QuantConfig &config);
  template <typename Type>
  void OperationForward();
  template <typename IType, typename WType>
  void OperationForward(x220::QuantConfig &config);
  std::shared_ptr<Tensor> input_;
  std::shared_ptr<Tensor> filter_;
  std::shared_ptr<Tensor> bias_;
  std::shared_ptr<Tensor> output_;
  std::shared_ptr<Tensor> data_workspace_;

  std::shared_ptr<Descriptor> convolution_;

  size_t h_size_, w_size_, hidden_filter_;
};
}  // namespace cpu

#endif  // CPU_OPS_CONVOLUTION_HPP
