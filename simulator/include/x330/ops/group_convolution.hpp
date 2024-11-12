#pragma once

#include <memory>

#include "inference_context.hpp"
#include "x330/ops/convolution.hpp"
#include "x330/x330_operation.hpp"

namespace x330 {
class GroupConvolution final : public Convolution {
 public:
  ~GroupConvolution() {}
  static std::unique_ptr<X330Operation> Create();
};
}  // namespace x330
