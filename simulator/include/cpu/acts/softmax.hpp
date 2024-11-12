#ifndef CPU_ACTS_SOFTMAX_HPP
#define CPU_ACTS_SOFTMAX_HPP

#include <memory>

#include "cpu/acts/activation.hpp"
#include "datatype.hpp"
#include "network/dimension.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cpu_operation.hpp"

namespace cpu {
class Softmax final : public Activation {
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
  template <typename Type>
  void ActivationForward();
};
}  // namespace cpu

#endif  // CPU_ACTS_SOFTMAX_HPP
