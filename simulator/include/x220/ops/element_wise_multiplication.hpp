#ifndef QUANTIZATION_X220_OPS_ELEMENT_WISE_MUL_HPP
#define QUANTIZATION_X220_OPS_ELEMENT_WISE_MUL_HPP

#include <memory>

#include "network/layer.hpp"
#include "network/network.hpp"
#include "x220/ops/x220_operation.hpp"

namespace x220 {
class EWMul : public X220Operation {
 public:
  EWMul();
  static std::unique_ptr<X220Operation> CreateQuantOperation();

 private:
  void InitQuantConfig(std::unique_ptr<Network> &network,
                       const int idx_layer) override;
  void QuantizeLayer(Layer &layer) override;
};
}  // namespace x220

#endif  // QUANTIZATION_X220_OPS_ELEMENT_WISE_MUL_HPP
