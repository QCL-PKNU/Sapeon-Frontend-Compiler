#ifndef CPU_ACTS_RELU6_HPP
#define CPU_ACTS_RELU6_HPP

#include <memory>

#include "cpu/acts/activation.hpp"
#include "datatype.hpp"
#include "network/dimension.hpp"
#include "network/layer.hpp"
#include "operations/cpu_operation.hpp"

namespace cpu {
class ReLU6 final : public Activation {
 public:
  static std::unique_ptr<CpuOperation> Create();
  bool CheckValidOperation(Layer& layer, Dimension input_dimension) override;
  Dimension CalculateOutputDimension(Layer& layer,
                                     Dimension input_dimension) override;

 private:
  void InitOutputTensor(dty::DataType dtype) override;
  void ActivationForward(Layer& layer) override;
  void ActivationQuantForward(Layer& layer) override;
};
}  // namespace cpu

#endif  // CPU_ACTS_RELU6_HPP
