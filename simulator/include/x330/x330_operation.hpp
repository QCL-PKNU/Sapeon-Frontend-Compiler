#pragma once

#include <memory>

#include "inference_context.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "x330/quant_config.hpp"

namespace x330 {
class X330Operation {
 public:
  virtual ~X330Operation() {}
  virtual void PrepareQuantOperation(std::unique_ptr<Network>& network,
                                     int idx_layer) = 0;
  virtual void Forward(Layer& layer, InferenceContext& ctx) = 0;

 protected:
  void ConvertInputTensor(std::shared_ptr<Tensor> tensor, QuantConfig& cfg);
  void ConvertActInTensor(std::shared_ptr<Tensor> tensor, QuantConfig& cfg);
  void ConvertOutputTensor(std::shared_ptr<Tensor> tensor, QuantConfig& cfg);
  void ConvertLayerFilter(Layer& layer);
};
}  // namespace x330
