#ifndef BACKENDS_DELEGATE_X220_INFERENCE_DELEGATE_HPP
#define BACKENDS_DELEGATE_X220_INFERENCE_DELEGATE_HPP

#include <memory>
#include <string>
#include <vector>

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "backends/delegate/inference_delegate.hpp"
#include "backends/delegate/inference_dump_helper.hpp"
#include "enums/error.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "tl/expected.hpp"

class X220InferenceDelegate : public InferenceDelegate {
 public:
  X220InferenceDelegate(Backend &parent, Arguments &args);
  tl::expected<void, SimulatorError> Inference(
      std::unique_ptr<Network> &network) override;

 private:
  InferenceDumpHelper dump_;
  Backend &parent_;
  std::string image_path_;
};

#endif  // BACKENDS_DELEGATE_X220_INFERENCE_DELEGATE_HPP
