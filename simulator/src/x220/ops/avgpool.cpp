#include "x220/ops/avgpool.hpp"

#define BASE X220Operation
#define NAME Avgpool
#define CLASS Avgpool
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

#include <cassert>
#include <memory>

#include "glog/logging.h"
using std::make_unique;
using std::unique_ptr;

#include "factory.hpp"
#include "network/layer.hpp"
#include "x220/ops/x220_operation.hpp"
#include "x220/quant_config.hpp"

namespace x220 {
static bool kRegistered = Factory<BASE>::RegisterCreateFunction(
    GET_STR(NAME), CLASS::CreateQuantOperation);

unique_ptr<BASE> SCOPE::CreateQuantOperation() { return make_unique<CLASS>(); }

SCOPE::Avgpool() {}

void SCOPE::InitQuantConfig(unique_ptr<Network> &network, const int idx_layer) {
  assert(idx_layer > 0);
  Layer &layer = network->layers(idx_layer);

  InitCommonQuantConfig(network, idx_layer);

  auto &config = layer.x220_quant_config();
  if (!x220::IsUnsigned(config.out_dtype())) {
    config.oqmin(config.oqmin() + 1);
  }

  const float iscale = config.iscale();
  const float oscale = config.oscale();

  // recalculate shifter & multiplier
  float threshold_ratio =
      oscale / iscale /
      float(layer.input_dimensions(0).w() * layer.input_dimensions(0).h());
  int multiplier, shifter;
  CalculateMultiplierShifter(config.in_dtype(), threshold_ratio, multiplier,
                             shifter);
  config.multiplier(multiplier);
  config.shifter(shifter);
}

void SCOPE::QuantizeLayer(Layer &layer) {}
}  // namespace x220
