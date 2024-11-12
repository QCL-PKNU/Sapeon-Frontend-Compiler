#ifndef CPU_OPS_ACTIVATIONS_HPP
#define CPU_OPS_ACTIVATIONS_HPP

#include <memory>

#include "datatype.hpp"
#include "inference_context.hpp"
#include "network/descriptor.hpp"
#include "network/dimension.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cpu_operation.hpp"

namespace cpu {
class Activations final : public CpuOperation {
 public:
  static std::unique_ptr<CpuOperation> Create();
  void Forward(Layer &layer, InferenceContext &ctx) override;
  bool CheckValidOperation(Layer &layer, Dimension input_dimension) override;
  Dimension CalculateOutputDimension(Layer &layer,
                                     Dimension input_dimension) override;

 private:
  std::shared_ptr<Tensor> input_;
  std::shared_ptr<Tensor> filter_;
  std::shared_ptr<Tensor> output_;
};
}  // namespace cpu

#endif  // CPU_OPS_ACTIVATIONS_HPP
