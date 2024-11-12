#ifndef BACKENDS_DELEGATE_X220_QUANTIZATION_DELEGATE_HPP
#define BACKENDS_DELEGATE_X220_QUANTIZATION_DELEGATE_HPP

#include <memory>
#include <set>
#include <string>

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "backends/delegate/quantization_delegate.hpp"
#include "backends/delegate/quantization_dump_helper.hpp"
#include "enums/error.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "tl/expected.hpp"

namespace quantization {
class X220QuantizationDelegate : public QuantizationDelegate {
 public:
  X220QuantizationDelegate(Backend &parent, Arguments &args);
  tl::expected<void, SimulatorError> Quantize(
      std::unique_ptr<Network> &network) override;

 private:
  void InitQuantConfigDataType(std::unique_ptr<Network> &network);
  void PromoteQuantConfigDataType(std::unique_ptr<Network> &network);
  void SetQuantConfig(std::unique_ptr<Network> &network);
  bool UpdateDataType(std::unique_ptr<Network> &network, int idx_layer);
  void TryPromotion(std::unique_ptr<Network> &network, int idx_layer);
  bool AllowOutputPromotion(std::set<int> &visited,
                            std::unique_ptr<Network> &network, Layer &layer);
  bool AllowInputPromotion(std::set<int> &visited,
                           std::unique_ptr<Network> &network, Layer &layer);
  bool AllowSuccessorInputPromotion(std::set<int> &visited,
                                    std::unique_ptr<Network> &network,
                                    Layer &layer);
  bool AllowPredecessorOutputPromotion(std::set<int> &visited,
                                       std::unique_ptr<Network> &network,
                                       Layer &layer);
  bool CheckConvLayer(Layer &layer);
  bool CheckConvOrConnected(Layer &layer);
  bool CheckNegSlope(Layer &layer);
  QuantizationDumpHelper dump_;
  Backend &parent_;
};
}  // namespace quantization

#endif  // BACKENDS_DELEGATE_X220_QUANTIZATION_DELEGATE_HPP
