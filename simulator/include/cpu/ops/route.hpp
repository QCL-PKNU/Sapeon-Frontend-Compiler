#ifndef CPU_OPS_ROUTE_HPP
#define CPU_OPS_ROUTE_HPP

#include <memory>
#include <vector>

#include "datatype.hpp"
#include "inference_context.hpp"
#include "network/descriptor.hpp"
#include "network/dimension.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cpu_operation.hpp"

namespace cpu {
class Route : public CpuOperation {
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
  size_t num_inputs_;
  size_t num_channels_;
  std::vector<std::shared_ptr<Tensor>> inputs_;
  std::shared_ptr<Tensor> output_;
};
}  // namespace cpu

#endif  // CPU_OPS_ROUTE_HPP
