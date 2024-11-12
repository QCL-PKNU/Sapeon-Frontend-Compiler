#include "x220/ops/bilinear_upsample.hpp"

#define BASE X220Operation
#define NAME BilinearUpsample
#define CLASS BilinearUpsample
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

#include "factory.hpp"
#include "glog/logging.h"

using std::make_unique;
using std::unique_ptr;

namespace x220 {
static bool kRegistered = Factory<BASE>::RegisterCreateFunction(
    GET_STR(NAME), CLASS::CreateQuantOperation);

unique_ptr<BASE> SCOPE::CreateQuantOperation() { return make_unique<CLASS>(); }

SCOPE::BilinearUpsample() {}

#ifndef CONFIDENTIAL_FEATURES
void SCOPE::InitQuantConfig(unique_ptr<Network>&, const int) {
  LOG(FATAL) << "Unsupported operation, please check your build configuration";
  exit(1);
}
void SCOPE::QuantizeLayer(Layer&) {
  LOG(FATAL) << "Unsupported operation, please check your build configuration";
  exit(1);
}
#endif  // CONFIDENTIAL_FEATURES

}  // namespace x220
