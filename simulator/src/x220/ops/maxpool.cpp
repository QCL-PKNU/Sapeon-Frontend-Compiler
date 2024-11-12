#include "x220/ops/maxpool.hpp"

#define BASE X220Operation
#define NAME Maxpool
#define CLASS Maxpool
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

#include <cassert>
#include <memory>
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

SCOPE::Maxpool() {}

#ifndef CONFIDENTIAL_FEATURES
void SCOPE::InitQuantConfig(unique_ptr<Network> &network, const int idx_layer) {
  assert(idx_layer > 0);
  Layer &layer = network->layers(idx_layer);

  InitCommonQuantConfig(network, idx_layer);

  auto &config = layer.x220_quant_config();
  Layer &prev_layer = network->layers(idx_layer - 1);
  auto &prev_config = prev_layer.x220_quant_config();

  config.oscale(prev_config.oscale());
  config.oqbias(prev_config.oqbias());
  config.oqmin(prev_config.oqmin());
  config.oqmax(prev_config.oqmax());
}

void SCOPE::QuantizeLayer(Layer &) {}
#endif  // CONFIDENTIAL_FEATURES
}  // namespace x220
