#pragma once

#include <memory>

#include "inference_context.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "x330/x330_operation.hpp"

namespace x330 {
class Convolution : public X330Operation {
 public:
  virtual ~Convolution() {}
  static std::unique_ptr<X330Operation> Create();
  void PrepareQuantOperation(std::unique_ptr<Network>& network,
                             int idx_layer) override;
  void Forward(Layer& layer, InferenceContext& ctx) override;
  std::shared_ptr<Tensor> ForwardMatMul(std::shared_ptr<Tensor> input,
                                        Layer& layer);
  void ForwardBiasAdd(std::shared_ptr<Tensor> tensor,
                      std::shared_ptr<Tensor> bias);
};
}  // namespace x330
