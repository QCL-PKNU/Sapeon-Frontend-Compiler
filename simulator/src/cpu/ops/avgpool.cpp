#include "cpu/ops/avgpool.hpp"

#define BASE CpuOperation
#define NAME Avgpool
#define CLASS cpu::NAME
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

#include <algorithm>
#include <memory>
#include <string>

#include "datatype.hpp"
#include "factory.hpp"
#include "glog/logging.h"
#include "inference_context.hpp"
#include "network/descriptor.hpp"
#include "network/dimension.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cpu_operation.hpp"
#include "utility.hpp"
#include "x220/quant_config.hpp"

static bool kRegistered =
    Factory<BASE>::RegisterCreateFunction(GET_STR(NAME), CLASS::Create);

std::unique_ptr<BASE> SCOPE::Create() { return std::make_unique<CLASS>(); }

void SCOPE::Forward(Layer &layer, InferenceContext &ctx) {
  input_ = ctx.InputTensor(0);
  const auto out_dtype = ctx.out_dtype();
  sampling_ = layer.sampling();

  if (sampling_->window_height() == 0 && sampling_->window_width() == 0) {
    sampling_->window_height(input_->h());
    sampling_->window_width(input_->w());
    sampling_->stride_height(1);
    sampling_->stride_width(1);
  }

  x220::QuantConfig &quant_config = layer.x220_quant_config();

  InitOutputTensor(out_dtype);
  OperationForward(quant_config);

  ctx.SetOutputTensor(output_);
}

bool SCOPE::CheckValidOperation(Layer &layer, Dimension input_dimension) {
  if (!layer.HasSamplingDescriptor()) {
    LOG(ERROR) << "Sampling descriptor not found";
    return false;
  }
  const int h = layer.sampling()->window_height();
  const int w = layer.sampling()->window_width();
  const int ih = layer.input_dimensions(0).h();
  const int iw = layer.input_dimensions(0).w();
  if (!(h == 0 && w == 0) && !(h == ih && w == iw)) {
    LOG(ERROR) << "Window height and width is different with input: "
               << "(input height, input width) = (" << ih << ", " << iw << ")"
               << "(window height, window width) = (" << h << ", " << w << ")";
    return false;
  }
  return true;
}

Dimension SCOPE::CalculateOutputDimension(Layer &layer,
                                          Dimension input_dimension) {
  size_t sh = layer.sampling()->stride_height();
  size_t sw = layer.sampling()->stride_width();
  size_t wh = layer.sampling()->window_height();
  size_t ww = layer.sampling()->window_width();
  size_t pht = layer.sampling()->padding_height_top();
  size_t phb = layer.sampling()->padding_height_bottom();
  size_t pwl = layer.sampling()->padding_width_left();
  size_t pwr = layer.sampling()->padding_width_right();

  float height = ((input_dimension.h() + (pht + phb) - wh) / sh) + 1;
  float width = ((input_dimension.w() + (pwl + pwr) - ww) / sw) + 1;

  return Dimension(input_dimension.n(), input_dimension.c(),
                   static_cast<int64_t>(height), static_cast<int64_t>(width));
}

void SCOPE::InitOutputTensor(dty::DataType dtype) {
  size_t sh = sampling_->stride_height();
  size_t sw = sampling_->stride_width();
  size_t wh = sampling_->window_height();
  size_t ww = sampling_->window_width();
  size_t pht = sampling_->padding_height_top();
  size_t phb = sampling_->padding_height_bottom();
  size_t pwl = sampling_->padding_width_left();
  size_t pwr = sampling_->padding_width_right();

  float height = ((input_->h() + (pht + phb) - wh) / sh) + 1;
  float width = ((input_->w() + (pwl + pwr) - ww) / sw) + 1;

  output_ = std::make_shared<Tensor>(input_->n(), input_->c(),
                                     static_cast<int>(height),
                                     static_cast<int>(width), dtype);
}

void SCOPE::OperationForward(x220::QuantConfig &config) {
  const dty::DataType itype = input_->dtype();
  const dty::DataType otype = output_->dtype();
  if (itype == dty::DataType::FP32 && otype == dty::DataType::FP32) {
    OperationForward<float>();
  } else if (itype == dty::DataType::FP64 && otype == dty::DataType::FP64) {
    OperationForward<double>();
  } else if (itype == dty::DataType::UINT8 && otype == dty::DataType::UINT8) {
    OperationQuantForward<uint8_t>(config);
  } else if (itype == dty::DataType::INT8 && otype == dty::DataType::INT8) {
    OperationQuantForward<int8_t>(config);
  } else if (itype == dty::DataType::INT16 && otype == dty::DataType::INT16) {
    OperationQuantForward<int16_t>(config);
  } else {
    DLOG(FATAL) << "avgpool is not implemented for: " << dty::NameOf(itype);
  }
}

template <typename Type>
void SCOPE::OperationForward() {
  Type *in_data = input_->data<Type>();
  Type *out_data = output_->data<Type>();

  const int nn = input_->n();
  const int cc = input_->c();
  const int hh = input_->h();
  const int ww = input_->w();

  for (int n = 0; n < nn; ++n) {
    for (int c = 0; c < cc; ++c) {
      const size_t out_index = c + n * cc;
      out_data[out_index] = 0;
      for (int i = 0; i < hh * ww; ++i) {
        const size_t in_index = i + hh * ww * out_index;
        out_data[out_index] += in_data[in_index];
      }
      out_data[out_index] /= hh * ww;
    }
  }
}

template <typename Type>
void SCOPE::OperationQuantForward(x220::QuantConfig &config) {
  // int odty = config.out_dtype();
  // auto mxc_bias = config.mxc_biases()[0];
  // auto mxc_scale = config.mxc_scales()[0];
  Type *in_data = input_->data<Type>();
  Type *out_data = output_->data<Type>();

  const int64_t oqmin = static_cast<int64_t>(config.oqmin());
  const int64_t oqmax = static_cast<int64_t>(config.oqmax());
  const int nn = input_->n();
  const int cc = input_->c();
  const int hh = input_->h();
  const int ww = input_->w();

  for (int n = 0; n < nn; ++n) {
    for (int c = 0; c < cc; ++c) {
      int64_t acc = 0;
      for (int i = 0; i < hh * ww; ++i) {
        const size_t in_index = i + hh * ww * (c + n * cc);
        acc += in_data[in_index] * config.multiplier();
      }
      long long bias =
          (config.shifter() >= 1) ? (1 << (config.shifter() - 1)) : 0;
      acc = (acc + bias) >> config.shifter();
      acc = std::max(std::min(acc, oqmax), oqmin);
      // acc += mxc_bias.field.bias;
      // int scaled = mxc_scale.Scale(acc, odty);

      const size_t out_index = c + n * cc;
      out_data[out_index] = acc;
    }
  }
}
