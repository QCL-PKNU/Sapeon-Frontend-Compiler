#include "x220/ops/activations.hpp"

#define BASE X220Operation
#define NAME Activations
#define CLASS NAME
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

#include <cassert>
using std::make_unique;
using std::unique_ptr;

#include "factory.hpp"

namespace x220 {

static bool kRegistered = Factory<BASE>::RegisterCreateFunction(
    GET_STR(NAME), CLASS::CreateQuantOperation);

unique_ptr<BASE> SCOPE::CreateQuantOperation() { return make_unique<CLASS>(); }

SCOPE::Activations() {}

#ifndef CONFIDENTIAL_FEATURES

void SCOPE::InitQuantConfig(unique_ptr<Network>& network, const int idx_layer) {
  InitCommonQuantConfig(network, idx_layer);
}

void SCOPE::QuantizeLayer(Layer& layer) {}

#endif  // CONFIDENTIAL_FEATURES

}  // namespace x220
