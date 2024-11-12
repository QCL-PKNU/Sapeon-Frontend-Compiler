#include "x330/ops/convolution.hpp"

#include <algorithm>
#include <memory>

#include "cpu/common/blas.hpp"
#include "cpu/common/gemm.hpp"
#include "cpu/common/im2col.hpp"
#include "factory.hpp"
#include "inference_context.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "x330/x330_operation.hpp"

namespace x330 {

static bool kRegistered = Factory<X330Operation>::RegisterCreateFunction(
    "Convolution", Convolution::Create);

std::unique_ptr<X330Operation> Convolution::Create() {
  return std::make_unique<Convolution>();
}

void Convolution::PrepareQuantOperation(std::unique_ptr<Network>& network,
                                        const int idx_layer) {
  auto& layer = network->layers(idx_layer);
  this->ConvertLayerFilter(layer);
}

void Convolution::Forward(Layer& layer, InferenceContext& ctx) {
  auto& cfg = layer.x330_quant_config();
  cfg.num_samples++;

  auto input = ctx.InputTensor(0);

  ConvertInputTensor(input, cfg);

  auto output = ForwardMatMul(input, layer);

  ConvertActInTensor(output, cfg);

  if (layer.HasBias()) {
    ForwardBiasAdd(output, layer.bias());
  }

  ConvertActInTensor(output, cfg);

  const auto& op_types = layer.operation_types();
  const bool activation_not_exists = std::find(op_types.begin(), op_types.end(),
                                               "Activations") == op_types.end();
  if (activation_not_exists) {
    ConvertOutputTensor(output, cfg);
  }

  ctx.SetOutputTensor(output);
}

std::shared_ptr<Tensor> Convolution::ForwardMatMul(
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

  const auto groups = layer.convolution()->groups();
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
      std::make_shared<Tensor>(in_n, filter_n, static_cast<size_t>(height),
                               static_cast<size_t>(width), input->dtype());
  auto* out_data = output->data<float>();
  const auto out_n = output->n();
  const auto out_c = output->c();
  const auto out_h = output->h();
  const auto out_w = output->w();
  const auto out_dim_size = output->dimension().size();
  std::fill(out_data, out_data + out_dim_size, 0.0F);

  const size_t workspace_size = filter_h * filter_w * out_h * out_w * in_c;
  auto workspace = std::make_shared<Tensor>(workspace_size * 2, input->dtype());

  const auto nweights = filter_n * filter_h * filter_w * in_c / groups;
  const auto m = filter_n / groups;
  const auto n = out_h * out_w;
  const auto k = nweights / filter_n;

  for (int i = 0; i < in_n; i++) {
    for (int j = 0; j < groups; j++) {
      auto* filter_pos = filter_data + j * nweights / groups;
      auto* workspace_data = workspace->data<float>();
      auto* out_pos = out_data + (i * groups + j) * n * m;
      auto* in_pos = in_data + (i * groups + j) * (in_c / groups) * in_h * in_w;
      cpu::im2col_cpu<float>(in_pos, in_c / groups, in_h, in_w, filter_h,
                             filter_w, pad_ht * dilation_h, pad_hb * dilation_h,
                             pad_wl * dilation_w, pad_wr * dilation_w, stride_h,
                             stride_w, dilation_h, dilation_w, workspace_data);
      cpu::Gemm<float>(0, 0, m, n, k, 1, filter_pos, k, workspace_data, n, 1,
                       out_pos, n);
    }
  }

  return output;
}

void Convolution::ForwardBiasAdd(std::shared_ptr<Tensor> tensor,
                                 std::shared_ptr<Tensor> bias) {
  float* bias_data = bias->data<float>();
  cpu::AddBias<float>(tensor->data<float>(), bias_data, tensor->n(),
                      tensor->c(), tensor->h() * tensor->w());
}
}  // namespace x330
