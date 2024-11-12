#include "x330/x330_operation.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

#include "glog/logging.h"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "x330/aixv_base.h"
#include "x330/aixv_float.h"
#include "x330/aixv_utils.h"

namespace x330 {
void X330Operation::ConvertLayerFilter(Layer& layer) {
  const auto& cfg = layer.x330_quant_config();
  const auto wcalib = cfg.weight_calib;
  const auto& output_dim = layer.output_dimension();
  const auto& input_dim = layer.input_dimensions(0);
  auto filter = layer.filter();
  // input_dim.c() == filter->c()
  // output_dim.c() == filter->n()
  const int filter_size = filter->c() * filter->h() * filter->w();

  if (wcalib != QuantConfig::WcalMode::WCAL_NONE) {
    std::vector<float> filter_max(output_dim.c());
    float layer_max = std::numeric_limits<float>::min();
    float* filter_data = filter->data<float>();

    for (int n = 0; n < filter->n(); ++n) {
      auto* filter = filter_data + n * filter_size;
      auto maxv = *std::max_element(
          filter, filter + filter_size,
          [](float x, float y) { return std::abs(x) < std::abs(y); });
      filter_max[n] = std::abs(maxv);
      layer_max = std::max(layer_max, std::abs(maxv));
    }

    for (int n = 0; n < filter->n(); ++n) {
      const float maxv = wcalib == QuantConfig::WcalMode::WCAL_LAYER
                             ? layer_max
                             : filter_max[n];
      const float base_max = GetDtypeMax(cfg.weight_dtype);
      const int extra_exp_bias =
          -std::ceil(std::log2(static_cast<double>(maxv) / base_max));
      const float new_max = GetDtypeMax(cfg.weight_dtype, extra_exp_bias);
      if (maxv > new_max) {
        LOG(ERROR) << maxv << " > " << new_max;
        exit(1);
      }

      auto* partial_filter = filter_data + n * filter_size;
      ConvertX330Data(cfg.weight_dtype, extra_exp_bias, cfg.weight_rmode,
                      partial_filter, filter_size);
    }
  } else {
    ConvertX330Data(cfg.weight_dtype, cfg.weight_ebias, cfg.weight_rmode,
                    filter->data<float>(), filter->dimension().size());
  }

  ConvertX330Data(cfg.bias_dtype, cfg.bias_ebias, cfg.bias_rmode,
                  layer.bias()->data<float>(),
                  layer.bias()->dimension().size());
}

void X330Operation::ConvertInputTensor(std::shared_ptr<Tensor> tensor,
                                       QuantConfig& cfg) {
  auto* in_data = tensor->data<float>();
  const auto in_dim_size = tensor->dimension().size();
  cfg.input_max += CalculateX330Max(in_data, in_dim_size);
  ConvertX330Data(cfg.input_dtype, cfg.input_ebias, cfg.input_rmode, in_data,
                  in_dim_size);
}

void X330Operation::ConvertActInTensor(std::shared_ptr<Tensor> tensor,
                                       QuantConfig& cfg) {
  auto* actin_data = tensor->data<float>();
  const auto actin_dim_size = tensor->dimension().size();
  cfg.actin_max += CalculateX330Max(actin_data, actin_dim_size);
  ConvertX330Data(cfg.actin_dtype, cfg.actin_ebias, cfg.actin_rmode, actin_data,
                  actin_dim_size);
}

void X330Operation::ConvertOutputTensor(std::shared_ptr<Tensor> tensor,
                                        QuantConfig& cfg) {
  auto* out_data = tensor->data<float>();
  const auto out_dim_size = tensor->dimension().size();
  cfg.output_max += CalculateX330Max(out_data, out_dim_size);
  ConvertX330Data(cfg.output_dtype, cfg.output_ebias, cfg.output_rmode,
                  out_data, out_dim_size);
}
}  // namespace x330
