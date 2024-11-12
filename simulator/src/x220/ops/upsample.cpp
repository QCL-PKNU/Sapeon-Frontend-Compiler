#include "x220/ops/upsample.hpp"

#define BASE X220Operation
#define NAME Upsample
#define CLASS Upsample
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

#include "factory.hpp"
#include "glog/logging.h"

namespace x220 {
static bool kRegistered = Factory<BASE>::RegisterCreateFunction(
    GET_STR(NAME), CLASS::CreateQuantOperation);

std::unique_ptr<BASE> SCOPE::CreateQuantOperation() {
  return std::make_unique<CLASS>();
}

SCOPE::Upsample() {}

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
