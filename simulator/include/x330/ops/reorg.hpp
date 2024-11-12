#pragma once

#include <memory>

#include "inference_context.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "x330/ops/x330_unary_operation.hpp"
#include "x330/x330_operation.hpp"

namespace x330 {
class Reorg final : public X330UnaryOperation {
 public:
  ~Reorg() {}
  static std::unique_ptr<X330Operation> Create();
  void PrepareQuantOperation(std::unique_ptr<Network>& network,
                             int idx_layer) override;
  void Forward(Layer& layer, InferenceContext& ctx) override;
};
}  // namespace x330
