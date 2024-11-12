#include "x220/ops/element_wise_multiplication.hpp"

#define BASE X220Operation
#define NAME EWMul
#define CLASS EWMul
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

#include <cassert>
#include <memory>

#include "factory.hpp"
#include "network/layer.hpp"
#include "x220/ops/x220_operation.hpp"

namespace x220 {
static bool kRegistered = Factory<BASE>::RegisterCreateFunction(
    GET_STR(NAME), CLASS::CreateQuantOperation);

std::unique_ptr<BASE> SCOPE::CreateQuantOperation() {
  return std::make_unique<CLASS>();
}

SCOPE::EWMul() {}

#ifndef CONFIDENTIAL_FEATURES
void SCOPE::InitQuantConfig(std::unique_ptr<Network>&, const int) {
  LOG(FATAL) << "Unsupported operation, please check your build configuration";
  exit(1);
}

void SCOPE::QuantizeLayer(Layer&) {
  LOG(FATAL) << "Unsupported operation, please check your build configuration";
  exit(1);
}
#endif  // CONFIDENTIAL_FEATURES
}  // namespace x220
