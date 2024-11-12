#pragma once

#include <memory>
#include <string>

#include "enums/error.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "tl/expected.hpp"
#include "x330/x330_operation.hpp"

namespace x330 {
class X330UnaryOperation : public X330Operation {
 public:
  virtual ~X330UnaryOperation() {}
  tl::expected<void, SimulatorError> ForwardUnaryOperation(
      Layer& layer, InferenceContext& ctx, const std::string& ops_name);
};
}  // namespace x330
