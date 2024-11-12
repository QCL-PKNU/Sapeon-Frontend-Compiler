#include "cpu/ops/connected.hpp"

#define BASE CpuOperation
#define NAME Connected
#define CLASS cpu::NAME
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

#include <memory>
#include <string>

#include "cpu/common/blas.hpp"
#include "cpu/common/gemm.hpp"
#include "cpu/common/im2col.hpp"
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
  auto out_dtype = ctx.out_dtype();
  filter_ = layer.filter();
  bias_ = layer.bias();
  convolution_ = layer.convolution();
  x220::QuantConfig &quant_config = layer.x220_quant_config();

  InitOutputTensor(out_dtype);
  AllocateMemory(out_dtype);
  OperationForward(quant_config);

  ctx.SetOutputTensor(output_);
}

bool SCOPE::CheckValidOperation(Layer &layer, Dimension input_dimension) {
  if (!layer.HasFilter()) {
    LOG(ERROR) << "Filter not found";
    return false;
  }
  if (!layer.HasConvolutionDescriptor()) {
    LOG(ERROR) << "Convolution descriptor not found";
    return false;
  }
  if (!layer.HasBias()) {
    return true;
  }
  if (layer.bias()->dimension().size() != layer.filter()->n()) {
    LOG(ERROR) << "Mismatch: bias size = " << layer.bias()->dimension().size()
               << ", number of filters = " << layer.filter()->n();
    return false;
  }
  return true;
}

Dimension SCOPE::CalculateOutputDimension(Layer &layer,
                                          Dimension input_dimension) {
  const size_t fh = layer.filter()->h();
  const size_t fw = layer.filter()->w();

  const size_t sh = layer.convolution()->stride_height();
  const size_t sw = layer.convolution()->stride_width();
  const size_t dh = layer.convolution()->dilation_height();
  const size_t dw = layer.convolution()->dilation_width();
  const size_t pht = layer.convolution()->padding_height_top();
  const size_t phb = layer.convolution()->padding_height_bottom();
  const size_t pwl = layer.convolution()->padding_width_left();
  const size_t pwr = layer.convolution()->padding_width_right();

  float height =
      ((input_dimension.h() + (pht + phb) - fh - (fh - 1) * (dh - 1)) / sh) + 1;
  float width =
      ((input_dimension.w() + (pwl + pwr) - fw - (fw - 1) * (dw - 1)) / sw) + 1;

  return Dimension(input_dimension.n(), layer.filter()->n(),
                   static_cast<int64_t>(height), static_cast<int64_t>(width));
}

void SCOPE::InitOutputTensor(dty::DataType dtype) {
  const size_t fh = filter_->h();
  const size_t fw = filter_->w();

  const size_t sh = convolution_->stride_height();
  const size_t sw = convolution_->stride_width();
  const size_t dh = convolution_->dilation_height();
  const size_t dw = convolution_->dilation_width();
  const size_t pht = convolution_->padding_height_top();
  const size_t phb = convolution_->padding_height_bottom();
  const size_t pwl = convolution_->padding_width_left();
  const size_t pwr = convolution_->padding_width_right();

  float height =
      ((input_->h() + (pht + phb) - fh - (fh - 1) * (dh - 1)) / sh) + 1;
  float width =
      ((input_->w() + (pwl + pwr) - fw - (fw - 1) * (dw - 1)) / sw) + 1;

  output_ = std::make_shared<Tensor>(input_->n(), filter_->n(),
                                     static_cast<int>(height),
                                     static_cast<int>(width), dtype);
}

void SCOPE::AllocateMemory(dty::DataType dtype) {
  const size_t workspace_size =
      filter_->h() * filter_->w() * output_->h() * output_->w() * input_->c();

  data_workspace_ =
      std::make_shared<Tensor>(workspace_size * 2, dtype);  // need to handle bT
}

void SCOPE::OperationForward(x220::QuantConfig &config) {
  const dty::DataType itype = input_->dtype();
  if (itype == dty::DataType::FP32) {
    OperationForward<float>();
  } else if (itype == dty::DataType::FP64) {
    OperationForward<double>();
  } else if (itype == dty::DataType::UINT8) {
    OperationForward<uint8_t, int8_t>(config);
  } else if (itype == dty::DataType::INT8) {
    OperationForward<int8_t, int8_t>(config);
  } else if (itype == dty::DataType::INT16) {
    OperationForward<int16_t, int16_t>(config);
  } else {
    DLOG(FATAL) << "connected is not implemented for: " << dty::NameOf(itype);
  }
}

template <typename Type>
void SCOPE::OperationForward() {
  Type *in_data = input_->data<Type>();
  Type *out_data = output_->data<Type>();
  Type *filter_data = filter_->data<Type>();

  const size_t input_size = input_->c() * input_->h() * input_->w();
  const size_t output_size = output_->c() * output_->h() * output_->w();

  for (int i = 0; i < output_->dimension().size(); ++i) {
    out_data[i] = static_cast<Type>(0);
  }

  Gemm<Type>(0, 1, input_->n(), output_size, input_size, 1, in_data, input_size,
             filter_data, input_size, 1, out_data, output_size);
  if (bias_ != nullptr) {
    Type *bias = bias_->data<Type>();
    AddBias<Type>(out_data, bias, output_->n(), output_->c(),
                  output_->h() * output_->w());
  }
}

// `forward_convolutional_layer_mxc_strict` from darknet_quant
template <typename IType, typename WType>
void SCOPE::OperationForward(x220::QuantConfig &config) {
  WType *weights = filter_->data<WType>();
  Dimension dims = output_->dimension();
  if (config.out_dtype() == x220::DataType::DTY_UINT8) {
    output_ = std::make_shared<Tensor>(dims.dims(), dty::DataType::UINT8);
  } else if (config.out_dtype() == x220::DataType::DTY_SINT8) {
    output_ = std::make_shared<Tensor>(dims.dims(), dty::DataType::SINT8);
  }

  // l.n = hidden_filter_
  // l.h = input_->h()
  // l.w = input_->w()
  // l.c = input_->c()
  // l.groups = convolution_.groups()
  // l.size = h_size_, w_size_
  // l.batch = input_->n()

  const size_t hidden_filter = filter_->dimension().dims()[0];
  const size_t h_size = filter_->h();
  const size_t w_size = filter_->w();

  const size_t groups = convolution_->groups();
  const size_t nweights =
      h_size * w_size * hidden_filter * input_->c() / groups;
  const size_t m = hidden_filter / groups;
  const size_t filter_size = h_size * w_size * input_->c() / groups;
  const size_t output_size = output_->h() * output_->w();

  const size_t stride_height = convolution_->stride_height();
  const size_t stride_width = convolution_->stride_width();
  const size_t dilation_height = convolution_->dilation_height();
  const size_t dilation_width = convolution_->dilation_width();
  const size_t padding_height = convolution_->padding_height_top();
  const size_t padding_width = convolution_->padding_width_left();

  for (int i = 0; i < input_->n(); ++i) {
    for (int j = 0; j < groups; ++j) {
      WType *a = weights + j * nweights / groups;
      IType *work_data = data_workspace_->data<IType>();

      // FIXME: is output dtype is always same with WType?
      WType *output_pos =
          output_->data<WType>() + (i * groups + j) * output_size * m;

      IType *im = input_->data<IType>() + (i * groups + j) * input_->h() *
                                              input_->w() *
                                              (input_->c() / groups);

      im2col_mxc<IType>(im, input_->c() / groups, input_->h(), input_->w(),
                        h_size, w_size, padding_height * dilation_height,
                        padding_width * dilation_width, stride_height,
                        stride_width, dilation_height, dilation_width,
                        work_data);

      IType *bT = work_data + output_size * filter_size;
      for (int nn = 0; nn < output_size; ++nn) {
        for (int kk = 0; kk < filter_size; ++kk) {
          bT[nn * filter_size + kk] = work_data[kk * output_size + nn];
        }
      }

      // #pragma omp parallel for
      for (int mm = 0; mm < m; ++mm) {
        // output channel
        auto odty = config.out_dtype();
        auto mxc_bias = config.mxc_biases()[mm];
        auto mxc_scale = config.mxc_scales()[mm];

        for (int nn = 0; nn < output_size; ++nn) {
          // output pixel
          int64_t acc = 0;
          for (int kk = 0; kk < filter_size; ++kk) {
            IType input = bT[nn * filter_size + kk];
            WType weight = a[mm * filter_size + kk];
            acc += (int)input * weight;
          }

          acc += mxc_bias.field.bias;
          int scaled = mxc_scale.Scale(acc, odty);
          output_pos[mm * output_size + nn] = scaled;
        }  // nn : output pixel
      }    // mm : output channel
    }      // j : group
  }        // i : batch
}
