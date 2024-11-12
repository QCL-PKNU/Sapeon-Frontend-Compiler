#ifndef CPU_OPS_CW_PRELU_HPP
#define CPU_OPS_CW_PRELU_HPP

#include <memory>

#include "cpu/acts/activation.hpp"
#include "datatype.hpp"
#include "network/layer.hpp"
#include "operations/cpu_operation.hpp"

namespace cpu {

class CWPrelu final : public Activation {
 public:
  static std::unique_ptr<CpuOperation> Create();
  bool CheckValidOperation(Layer& layer, Dimension input_dimension) override;
  Dimension CalculateOutputDimension(Layer& layer,
                                     Dimension input_dimension) override;

 private:
  void InitOutputTensor(dty::DataType dtype) override {
    Activation::InitOutputTensor(dtype);
  }
  void ActivationForward(Layer& layer) override;
  void ActivationQuantForward(Layer& layer) override;
  template <typename T>
  void ActivationForward(const std::vector<float>& negative_slope);
};
}  // namespace cpu

#endif  // CPU_OPS_CW_PRELU_HPP
