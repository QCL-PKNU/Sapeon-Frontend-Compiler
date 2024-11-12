#ifndef QUANTIZATION_X220_OPS_X220_OPERATION_HPP
#define QUANTIZATION_X220_OPS_X220_OPERATION_HPP

#include <memory>

#include "network/layer.hpp"
#include "network/network.hpp"
#include "x220/quant_config.hpp"

namespace x220 {
class X220Operation {
 public:
  void PrepareQuantOperation(std::unique_ptr<Network>& network, int idx_layer);
  static void InitCommonQuantConfig(std::unique_ptr<Network>& network,
                                    int idx_layer);
  virtual ~X220Operation() {}

 protected:
  virtual void InitQuantConfig(std::unique_ptr<Network>& network,
                               int idx_layer) = 0;
  virtual void QuantizeLayer(Layer& layer) = 0;
  static void CalculateMultiplierShifter(DataType dtype, float threshold_ratio,
                                         int& multiplier, int& shifter);
};
}  // namespace x220

#endif  // QUANTIZATION_X220_OPS_X220_OPERATION_HPP
