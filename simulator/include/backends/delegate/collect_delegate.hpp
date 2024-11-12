#pragma once

#include <memory>
#include <string>

#include "enums/error.hpp"
#include "network/network.hpp"
#include "tl/expected.hpp"

namespace collect {
class CollectDelegate {
 public:
  virtual tl::expected<void, SimulatorError> Collect(
      std::unique_ptr<Network> &network) = 0;
  virtual ~CollectDelegate() {}
};
}  // namespace collect
