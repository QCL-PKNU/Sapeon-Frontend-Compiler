#pragma once

#include <memory>
#include <string>

#include "enums/error.hpp"
#include "network/network.hpp"
#include "tl/expected.hpp"

namespace quantization {
class QuantizationDelegate {
 public:
  virtual tl::expected<void, SimulatorError> Quantize(
      std::unique_ptr<Network> &network) = 0;
  virtual ~QuantizationDelegate() {}
};
}  // namespace quantization
