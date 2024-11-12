#include "cpu/acts/cw_prelu.hpp"

#define BASE CpuOperation
#define NAME CWPrelu
#define CLASS cpu::NAME
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

#include <cassert>
#include <memory>
using std::make_unique;
using std::unique_ptr;

#include "factory.hpp"
#include "glog/logging.h"

using dty::DataType;
using x220::QuantConfig;

static bool kRegistered =
    Factory<BASE>::RegisterCreateFunction(GET_STR(NAME), CLASS::Create);

std::unique_ptr<BASE> SCOPE::Create() { return std::make_unique<CLASS>(); }

bool SCOPE::CheckValidOperation(Layer &layer, Dimension input_dimension) {
  return true;
};

Dimension SCOPE::CalculateOutputDimension(Layer &layer,
                                          Dimension input_dimension) {
  return input_dimension;
}

void SCOPE::ActivationForward(Layer &layer) {
  const auto itype = input_->dtype();
  const auto otype = output_->dtype();
  const auto &negative_slope = layer.negative_slope();

  if (itype == dty::DataType::FP32 && otype == dty::DataType::FP32) {
    ActivationForward<float>(negative_slope);
  } else if (itype == dty::DataType::FP64 && otype == dty::DataType::FP64) {
    ActivationForward<double>(negative_slope);
  } else {
    LOG(ERROR) << "Not implemented forward type: itype=" << dty::NameOf(itype)
               << ", otype=" << dty::NameOf(otype) << '\n';
    exit(1);
  }
}

template <typename Type>
void SCOPE::ActivationForward(const std::vector<float> &negative_slope) {
  Type *input_data = input_->data<Type>();
  Type *output_data = output_->data<Type>();

  for (int n = 0; n < input_->n(); ++n) {
    for (int c = 0; c < input_->c(); ++c) {
      // Channel-wise alpha coefficient
      Type alpha_val = negative_slope[c];

      for (int h = 0; h < input_->h(); ++h) {
        for (int w = 0; w < input_->w(); ++w) {
          int input_idx =
              ((n * input_->c() + c) * input_->h() + h) * input_->w() + w;
          Type input_val = input_data[input_idx];

          // Apply PReLU activation function
          output_data[input_idx] =
              (input_val > 0) ? input_val : alpha_val * input_val;
        }
      }
    }
  }
}

#ifndef CONFIDENTIAL_FEATURES
void SCOPE::ActivationQuantForward(Layer &) {
  LOG(ERROR) << "Not implemented forward type" << '\n';
  exit(1);
}
#endif
