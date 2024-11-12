#ifndef BACKENDS_DELEGATE_INFERENCE_DELEGATE_HPP
#define BACKENDS_DELEGATE_INFERENCE_DELEGATE_HPP

#include <memory>
#include <string>

#include "enums/error.hpp"
#include "network/network.hpp"
#include "tl/expected.hpp"

class InferenceDelegate {
 public:
  virtual tl::expected<void, SimulatorError> Inference(
      std::unique_ptr<Network> &network) = 0;
  virtual ~InferenceDelegate() {}
};

#endif  // BACKENDS_DELEGATE_INFERENCE_DELEGATE_HPP
