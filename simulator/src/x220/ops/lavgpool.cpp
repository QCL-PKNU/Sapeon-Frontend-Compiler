#include "x220/ops/lavgpool.hpp"

#define BASE X220Operation
#define NAME Lavgpool
#define CLASS Lavgpool
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

SCOPE::Lavgpool() {}

#ifndef CONFIDENTIAL_FEATURES
void SCOPE::InitQuantConfig(unique_ptr<Network> &, const int) {
  LOG(FATAL) << "Unsupported operation, please check your build configuration";
  exit(1);
}
void SCOPE::QuantizeLayer(Layer &) {
  LOG(FATAL) << "Unsupported operation, please check your build configuration";
  exit(1);
}
#endif  // CONFIDENTIAL_FEATURES
}  // namespace x220
