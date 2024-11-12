#pragma once

#include <memory>

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "backends/delegate/collect_delegate.hpp"
#include "enums/error.hpp"
#include "network/network.hpp"
#include "tl/expected.hpp"

namespace collect {
class DummyCollectDelegate final : public CollectDelegate {
 public:
  DummyCollectDelegate(Backend &parent, Arguments &args) {}
  tl::expected<void, SimulatorError> Collect(
      std::unique_ptr<Network> &network) override {
    return {};
  }
};
}  // namespace collect
