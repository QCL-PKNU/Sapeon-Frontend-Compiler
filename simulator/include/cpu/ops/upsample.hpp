#ifndef CPU_OPS_UPSAMPLE_HPP
#define CPU_OPS_UPSAMPLE_HPP

#include <memory>

#include "datatype.hpp"
#include "inference_context.hpp"
#include "network/descriptor.hpp"
#include "network/dimension.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cpu_operation.hpp"

namespace cpu {
class Upsample final : public CpuOperation {
 public:
  static std::unique_ptr<CpuOperation> Create();
  void Forward(Layer &layer, InferenceContext &ctx) override;
  bool CheckValidOperation(Layer &layer, Dimension input_dimension) override;
  Dimension CalculateOutputDimension(Layer &layer,
                                     Dimension input_dimension) override;

 private:
  void InitOutputTensor(dty::DataType dtype);
  void OperationForward();
  template <typename Type>
  void OperationForward();
  std::shared_ptr<Tensor> input_;
  std::shared_ptr<Tensor> output_;
  std::shared_ptr<Descriptor> sampling_;
};
}  // namespace cpu

#endif  // CPU_OPS_UPSAMPLE_HPP
