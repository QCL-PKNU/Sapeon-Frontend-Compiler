#include "x330/ops/connected.hpp"

#include <algorithm>
#include <memory>

#include "cpu/common/blas.hpp"
#include "cpu/common/gemm.hpp"
#include "factory.hpp"
#include "inference_context.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"

namespace x330 {

static bool kRegistered = Factory<X330Operation>::RegisterCreateFunction(
    "Connected", Connected::Create);

std::unique_ptr<X330Operation> Connected::Create() {
  return std::make_unique<Connected>();
}

void Connected::PrepareQuantOperation(std::unique_ptr<Network>& network,
                                      const int idx_layer) {
  auto& layer = network->layers(idx_layer);
  this->ConvertLayerFilter(layer);
}

void Connected::Forward(Layer& layer, InferenceContext& ctx) {
  auto& cfg = layer.x330_quant_config();
  cfg.num_samples++;

  auto input = ctx.InputTensor(0);

  ConvertInputTensor(input, cfg);

  auto output = ForwardConnected(input, layer);

  ConvertActInTensor(output, cfg);

  const auto& op_types = layer.operation_types();
  const bool activation_not_exists = std::find(op_types.begin(), op_types.end(),
                                               "Activations") == op_types.end();
  if (activation_not_exists) {
    ConvertOutputTensor(output, cfg);
  }

  ctx.SetOutputTensor(output);
}

std::shared_ptr<Tensor> Connected::ForwardConnected(
    std::shared_ptr<Tensor> input, Layer& layer) {
  auto* in_data = input->data<float>();
  const auto in_dim_size = input->dimension().size();
  const auto in_n = input->n();
  const auto in_c = input->c();
  const auto in_h = input->h();
  const auto in_w = input->w();

  const auto filter_n = layer.filter()->n();
  const auto filter_h = layer.filter()->h();
  const auto filter_w = layer.filter()->w();
  auto* filter_data = layer.filter()->data<float>();

  const auto stride_h = layer.convolution()->stride_height();
  const auto stride_w = layer.convolution()->stride_width();
  const auto dilation_h = layer.convolution()->dilation_height();
  const auto dilation_w = layer.convolution()->dilation_width();
  const auto pad_ht = layer.convolution()->padding_height_top();
  const auto pad_hb = layer.convolution()->padding_height_bottom();
  const auto pad_wl = layer.convolution()->padding_width_left();
  const auto pad_wr = layer.convolution()->padding_width_right();

  const float height = (static_cast<float>(in_h + (pad_ht + pad_hb) - filter_h -
                                           (filter_h - 1) * (dilation_h - 1)) /
                        stride_h) +
                       1;
  const float width = (static_cast<float>(in_w + (pad_wl + pad_wr) - filter_w -
                                          (filter_w - 1) * (dilation_w - 1)) /
                       stride_w) +
                      1;

  auto output =
      std::make_shared<Tensor>(in_n, filter_n, static_cast<int64_t>(height),
                               static_cast<int64_t>(width), input->dtype());
  auto* out_data = output->data<float>();
  const auto out_n = output->n();
  const auto out_c = output->c();
  const auto out_h = output->h();
  const auto out_w = output->w();
  const auto out_dim_size = output->dimension().size();

  std::fill(out_data, out_data + out_dim_size, 0.0F);

  const auto input_size = in_c * in_h * in_w;
  const auto output_size = out_c * out_h * out_w;
  cpu::Gemm<float>(0, 1, in_n, output_size, input_size, 1, in_data, input_size,
                   filter_data, input_size, 1, out_data, output_size);
  if (layer.HasBias()) {
    float* bias = layer.bias()->data<float>();
    cpu::AddBias<float>(out_data, bias, out_n, out_c, out_h * out_w);
  }

  return output;
}
}  // namespace x330
