#include "x220/ops/element_wise_add.hpp"

#define BASE X220Operation
#define NAME EWAdd
#define CLASS EWAdd
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

SCOPE::EWAdd() {}

#ifndef CONFIDENTIAL_FEATURES
void SCOPE::InitQuantConfig(unique_ptr<Network>& network, const int idx_layer) {
  assert(idx_layer > 0);
  InitCommonQuantConfig(network, idx_layer);

  Layer& layer = network->layers(idx_layer);

  assert(layer.predecessors().size() == 2);
  const int a_index = layer.predecessors(0);
  const int b_index = layer.predecessors(1);

  Layer& a_layer = network->layers(a_index);
  Layer& b_layer = network->layers(b_index);

  auto& config = layer.x220_quant_config();
  auto& a_config = a_layer.x220_quant_config();
  auto& b_config = b_layer.x220_quant_config();

  const float a_scale = a_config.oscale();
  const float b_scale = b_config.oscale();
  const float o_scale = config.oscale();
  const float a_qbias = a_config.oqbias();
  const float b_qbias = b_config.oqbias();
  const float o_qbias = config.oqbias();
  const auto o_dtype = config.out_dtype();

  float a_mul_f = o_scale / a_scale;
  float b_mul_f = o_scale / b_scale;

  int fixed_min, fixed_max;
  x220::GetDataTypeMinMax(fixed_min, fixed_max, o_dtype);

  int shift_amt_a, fixed_A, shift_amt_b, fixed_B;
  CalculateMultiplierShifter(a_config.out_dtype(), a_mul_f, fixed_A,
                             shift_amt_a);
  CalculateMultiplierShifter(b_config.out_dtype(), b_mul_f, fixed_B,
                             shift_amt_b);

  // makeSameShift
  {
    int* shift0 = &shift_amt_a;
    int* fixed_R0 = &fixed_A;
    int* shift1 = &shift_amt_b;
    int* fixed_R1 = &fixed_B;
    int *larger = nullptr, *smaller = nullptr, *fixed_smaller = nullptr,
        *fixed_larger = nullptr;

    if (*shift0 >= *shift1) {
      larger = shift0;
      smaller = shift1;
      fixed_larger = fixed_R0;
      fixed_smaller = fixed_R1;
    } else {
      larger = shift1;
      smaller = shift0;
      fixed_larger = fixed_R1;
      fixed_smaller = fixed_R0;
    }

    while (*larger != *smaller && *fixed_larger / 2 >= fixed_min) {
      (*larger)--;
      *fixed_larger >>= 1;
    }
    if (*larger != *smaller) {
      while (*larger != *smaller && *fixed_smaller * 2 >= fixed_max) {
        (*smaller)++;
        *fixed_smaller <<= 1;
      }
    }
  }

  x220::QuantConfig::Shortcut shortcut;
  shortcut.a_mul = fixed_A;
  shortcut.b_mul = fixed_B;
  shortcut.rsh = shift_amt_a;

  config.shortcut(std::move(shortcut));
}

void SCOPE::QuantizeLayer(Layer&) {}
#endif  // CONFIDENTIAL_FEATURES
}  // namespace x220
