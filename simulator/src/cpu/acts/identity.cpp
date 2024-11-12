#include "cpu/acts/identity.hpp"

#define BASE CpuOperation
#define NAME Identity
#define CLASS cpu::NAME
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

#include <memory>
using std::make_unique;
using std::shared_ptr;
using std::unique_ptr;

#include "datatype.hpp"
using dty::DataType;
#include "factory.hpp"
#include "glog/logging.h"
#include "network/dimension.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cpu_operation.hpp"
#include "utility.hpp"

static bool kRegistered =
    Factory<BASE>::RegisterCreateFunction(GET_STR(NAME), CLASS::Create);

std::unique_ptr<BASE> SCOPE::Create() { return std::make_unique<CLASS>(); }

bool SCOPE::CheckValidOperation(Layer& layer, Dimension input_dimension) {
  return true;
};

Dimension SCOPE::CalculateOutputDimension(Layer& layer,
                                          Dimension input_dimension) {
  return input_dimension;
}

void SCOPE::ActivationForward(Layer& layer) {
  output_ = layer.intermediate_activation() == nullptr
                ? layer.inputs(0)
                : layer.intermediate_activation();
}

void SCOPE::ActivationQuantForward(Layer& layer) { ActivationForward(layer); }
