#pragma once

#include <memory>

#include "inference_context.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "x330/x330_operation.hpp"

namespace x330 {
class Connected final : public X330Operation {
 public:
  ~Connected() {}
  static std::unique_ptr<X330Operation> Create();
  void PrepareQuantOperation(std::unique_ptr<Network>& network,
                             int idx_layer) override;
  void Forward(Layer& layer, InferenceContext& ctx) override;
  std::shared_ptr<Tensor> ForwardConnected(std::shared_ptr<Tensor> input,
                                           Layer& layer);
};
}  // namespace x330
