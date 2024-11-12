#ifndef BACKENDS_DELEGATE_DUMMY_QUANTIZATION_DELEGATE_HPP
#define BACKENDS_DELEGATE_DUMMY_QUANTIZATION_DELEGATE_HPP

#include <memory>

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "backends/delegate/quantization_delegate.hpp"
#include "enums/error.hpp"
#include "network/network.hpp"
#include "tl/expected.hpp"

namespace quantization {
class DummyQuantizationDelegate : public QuantizationDelegate {
 public:
  DummyQuantizationDelegate(Backend &parent, Arguments &args);
  tl::expected<void, SimulatorError> Quantize(
      std::unique_ptr<Network> &network) override;
};
}  // namespace quantization

#endif  // BACKENDS_DELEGATE_DUMMY_QUANTIZATION_DELEGATE_HPP
