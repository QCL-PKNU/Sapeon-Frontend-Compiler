#include "x330/ops/group_convolution.hpp"

#include <memory>

#include "factory.hpp"
#include "x330/x330_operation.hpp"

namespace x330 {

constexpr auto kOpsName = "GroupConvolution";

static bool kRegistered = Factory<X330Operation>::RegisterCreateFunction(
    kOpsName, GroupConvolution::Create);

std::unique_ptr<X330Operation> GroupConvolution::Create() {
  return std::make_unique<GroupConvolution>();
}
}  // namespace x330
