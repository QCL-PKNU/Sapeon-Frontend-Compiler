#include "cpu/acts/softmax.hpp"

#define BASE CpuOperation
#define NAME Softmax
#define CLASS cpu::NAME
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

#include <cmath>
using std::exp;
#include <memory>
using std::make_unique;
using std::unique_ptr;

#include "datatype.hpp"
using dty::DataType;
#include "factory.hpp"
#include "glog/logging.h"
#include "network/dimension.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cpu_operation.hpp"

static bool kRegistered =
    Factory<BASE>::RegisterCreateFunction(GET_STR(NAME), CLASS::Create);

std::unique_ptr<BASE> SCOPE::Create() { return std::make_unique<CLASS>(); }

bool SCOPE::CheckValidOperation(Layer &layer, Dimension input_dimension) {
  return true;
}

Dimension SCOPE::CalculateOutputDimension(Layer &layer,
                                          Dimension input_dimension) {
  return input_dimension;
}

void SCOPE::ActivationForward(Layer &) {
  const dty::DataType itype = input_->dtype();
  const dty::DataType otype = output_->dtype();
  if (itype == dty::DataType::FP32 && otype == dty::DataType::FP32) {
    ActivationForward<float>();
  } else if (itype == dty::DataType::FP64 && otype == dty::DataType::FP64) {
    ActivationForward<double>();
  } else {
    DLOG(FATAL) << "softmax is not implemented for: " << dty::NameOf(itype);
  }
}

void SCOPE::ActivationQuantForward(Layer &) {
  LOG(ERROR) << "Not implemented forward type\n";
  exit(1);
}

template <typename Type>
void SCOPE::ActivationForward() {
  Type *in_data = input_->data<Type>();
  Type *out_data = output_->data<Type>();

  const int each_height_size = output_->w();
  const int each_channel_size = output_->h() * each_height_size;
  const int each_batch_size = output_->c() * each_channel_size;
  const int num_datas = output_->n() * each_batch_size;

  Type sum = 0;
  Type largest = std::numeric_limits<Type>::lowest();

  for (int n = 0; n < output_->n(); ++n)
    for (int c = 0; c < output_->c(); ++c)
      for (int h = 0; h < output_->h(); ++h)
        for (int w = 0; w < output_->w(); ++w) {
          int idx = n * each_batch_size + c * each_channel_size +
                    h * each_height_size + w;

          if (in_data[idx] > largest) largest = in_data[idx];
        }

  for (int n = 0; n < output_->n(); ++n)
    for (int c = 0; c < output_->c(); ++c)
      for (int h = 0; h < output_->h(); ++h)
        for (int w = 0; w < output_->w(); ++w) {
          int idx = n * each_batch_size + c * each_channel_size +
                    h * each_height_size + w;

          Type e = exp(in_data[idx] - largest);
          sum += e;
          out_data[idx] = e;
        }

  for (int n = 0; n < output_->n(); ++n)
    for (int c = 0; c < output_->c(); ++c)
      for (int h = 0; h < output_->h(); ++h)
        for (int w = 0; w < output_->w(); ++w) {
          int idx = n * each_batch_size + c * each_channel_size +
                    h * each_height_size + w;

          out_data[idx] /= sum;
        }
}
