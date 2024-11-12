#include "x220/ops/connected.hpp"

#define BASE X220Operation
#define NAME Connected
#define CLASS Connected
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
using std::make_shared;
using std::make_unique;
using std::shared_ptr;
using std::unique_ptr;
#include <vector>
using std::vector;

#include "factory.hpp"
#include "glog/logging.h"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "x220/ops/x220_operation.hpp"

namespace x220 {
static bool kRegistered = Factory<BASE>::RegisterCreateFunction(
    GET_STR(NAME), CLASS::CreateQuantOperation);

unique_ptr<BASE> SCOPE::CreateQuantOperation() { return make_unique<CLASS>(); }

SCOPE::Connected() {}

void SCOPE::InitQuantConfig(unique_ptr<Network> &network, const int idx_layer) {
  InitCommonQuantConfig(network, idx_layer);
  Layer &layer = network->layers(idx_layer);
  auto &config = layer.x220_quant_config();
  if (!x220::IsUnsigned(config.out_dtype())) {
    config.oqmin(config.oqmin() + 1);
  }

  const size_t output_c = layer.filter()->n();           // l.out_c
  const size_t input_c = layer.input_dimensions(0).c();  // l.c
  const size_t filter_h = layer.filter()->h();           // l.size
  const size_t filter_w = layer.filter()->w();           // l.size
  const size_t filter_size = input_c * filter_h * filter_w;

  auto quant_weights = vector<float>(output_c * filter_size);
  auto quant_biases = vector<float>(output_c);

  auto mxc_biases = vector<x220::MxcBias>(output_c);
  auto mxc_scales = vector<x220::MxcScale>(output_c);

  auto idty = config.in_dtype();
  auto iscale = config.iscale();
  auto odty = config.out_dtype();
  auto oscale = config.oscale();
  auto oqbias = config.oqbias();

  shared_ptr<Tensor> mxc_weights = std::make_shared<Tensor>(
      output_c, input_c, filter_h, filter_w,
      (idty == x220::DataType::DTY_INT16 ? dty::DataType::INT16
                                         : dty::DataType::INT8));

  // Dump weight scale
  char path_wscale[256];  // fp4
  sprintf(path_wscale, "./dump/wscale_%03d.txt", idx_layer);
  FILE *fp_wscale = fopen(path_wscale, "w");

  char path_mxcscale[256];  // fp7
  sprintf(path_mxcscale, "./dump/mxcscale_%03d.txt", idx_layer);
  FILE *fp_mxcscale = fopen(path_mxcscale, "w");

  // Dump weight threshold
  char path_wth[256];  // fp6
  sprintf(path_wth, "./dump/wth_%03d.txt", idx_layer);
  FILE *fp_wth = fopen(path_wth, "w");

  int wquant_max = (idty == x220::DataType::DTY_INT16) ? INT16_MAX : INT8_MAX;
  wquant_max *= config.wamplifier();

  for (int f = 0; f < output_c; ++f) {
    // TODO: what happens if dtype is FP64?
    assert(layer.filter()->dtype() == dty::DataType::FP32);
    float *iptr = layer.filter()->data<float>() + f * filter_size;
    float ub = static_cast<float>(INT8_MAX);
    x220::S64 ub_bias = x220::S64((0x1LL << (48 - 1)) - 1);
    float fthreshold = 0.0f;
    for (int i = 0; i < filter_size; ++i) {
      fthreshold = std::max(fthreshold, fabs(iptr[i]));
    }
    fprintf(fp_wth, "output[%d] = %.40f\n", f, fthreshold);  // wth: OK

    assert(layer.bias()->dtype() == dty::DataType::FP32);
    float new_bias = layer.bias()->data<float>()[f];
    float wscale = ub / fthreshold;
    fprintf(fp_wscale, "output[%d] = %.30lf\n", f, wscale);  // wscale: OK

    x220::S64 fixed_bias = x220::S64(std::round(new_bias * iscale * wscale));
    if (fixed_bias <= ub_bias && fixed_bias >= -ub_bias) {
      wscale = x220::MxcScale::AdjustWeightScale(wscale, iscale, oscale);
    } else {
      float bscale = float(ub_bias / 2) / fabs(new_bias) / iscale;
      wscale = x220::MxcScale::AdjustWeightScale(bscale, iscale, oscale);
    }
    fprintf(fp_mxcscale, "output[%d] = %.30lf\n", f, wscale);  // mxcscale: Diff

    for (int i = 0; i < filter_size; ++i) {
      float scaled = std::round(iptr[i] * wscale);
      // config.weights()[f * filter_size + i] =
      quant_weights.at(f * filter_size + i) =
          std::max(std::min(scaled, ub), -ub);
    }
    fixed_bias = x220::S64(std::round(new_bias * iscale * wscale));

    float neg_slope = 0.0;
    if (layer.activation_type() == "Identity") {
      neg_slope = 1.0;
    }
    mxc_scales.at(f).Encode(wscale, iscale, oscale, neg_slope);
    quant_biases.at(f) = 0.0;
    mxc_biases.at(f) = x220::MxcBias(fixed_bias);
  }
  fclose(fp_wscale);
  fclose(fp_mxcscale);
  fclose(fp_wth);

  config.weights(quant_weights);
  config.biases(quant_biases);
  config.mxc_scales(mxc_scales);
  config.mxc_biases(mxc_biases);

  for (int f = 0; f < output_c; ++f) {
    if (idty == x220::DataType::DTY_INT16) {
      auto optr = mxc_weights->data<int16_t>() + f * filter_size;
      for (int i = 0; i < filter_size; ++i) {
        optr[i] = config.weights()[f * filter_size + i];
      }
    } else {
      auto optr = mxc_weights->data<int8_t>() + f * filter_size;
      for (int i = 0; i < filter_size; ++i) {
        optr[i] = config.weights()[f * filter_size + i];
      }
    }
  }

  // Dump
  FILE *fp;
  char path[256];

  // Dump host weights
  sprintf(path, "./dump/fp_weight_%03d.txt", idx_layer);
  fp = fopen(path, "w");

  int host_fystride = filter_w;
  int host_fcstride = filter_h * filter_w;
  int host_fsize = input_c * filter_h * filter_w;

  for (int f = 0; f < output_c; ++f) {
    for (int c = 0; c < input_c; ++c) {
      for (int y = 0; y < filter_h; ++y) {
        for (int x = 0; x < filter_w; ++x) {
          int index =
              f * host_fsize + c * host_fcstride + y * host_fystride + x;
          // fprintf(fp, "output[%d] = %.40f\n", index,
          //         layer.filter()->data<float>()[index]);
          fprintf(fp, "output[%d] = ", index);
          union {
            float f;
            uint32_t u;
          } u;
          u.f = layer.filter()->data<float>()[index];
          fprintf(fp, "%02X ", (u.u & 0xff000000) >> 24);
          fprintf(fp, "%02X ", (u.u & 0xff0000) >> 16);
          fprintf(fp, "%02X ", (u.u & 0xff00) >> 8);
          fprintf(fp, "%02X\n", u.u & 0xff);
        }
      }
    }
  }
  fclose(fp);

  // Dump host biases
  sprintf(path, "./dump/fp_bias_%03d.txt", idx_layer);
  fp = fopen(path, "w");
  for (int f = 0; f < output_c; ++f) {
    // fprintf(fp, "output[%d] = %.50f\n", f, layer.bias()->data<float>()[f]);
    fprintf(fp, "output[%d] = ", f);
    union {
      float f;
      uint32_t u;
    } u;
    u.f = layer.bias()->data<float>()[f];
    fprintf(fp, "%02X ", (u.u & 0xff000000) >> 24);
    fprintf(fp, "%02X ", (u.u & 0xff0000) >> 16);
    fprintf(fp, "%02X ", (u.u & 0xff00) >> 8);
    fprintf(fp, "%02X\n", u.u & 0xff);
  }
  fclose(fp);

  // Dump quantized weights
  sprintf(path, "./dump/layer%03d.weight", idx_layer);
  fp = fopen(path, "w");

  for (int f = 0; f < output_c; ++f) {
    int64_t weight_sum = 0;
    if (idty == x220::DataType::DTY_INT16) {
      int depth = x220::IDivCeil(input_c, 4);
      auto get_element = [&](int x, int y, int z) -> int16_t {
        auto base = mxc_weights->data<int16_t>() + f * filter_size;
        if (z >= input_c) {
          return 0;
        }
        int idx = filter_h * filter_w * z + filter_w * y + x;
        return base[idx];
      };

      for (int xy = 0; xy < filter_h * filter_w; ++xy) {
        int x = xy % filter_w;
        int y = xy / filter_w;
        for (int z = 0; z < depth; ++z) {
          uint64_t word = 0;
          for (int i = 0; i < 4; ++i) {
            auto elem = get_element(x, y, 4 * z + i);
            weight_sum += elem;
            word |= uint64_t(uint16_t(elem)) << (16 * i);
          }
          fprintf(fp, "%016lX ", word);
        }
      }  // kxy
    } else {
      int depth = x220::IDivCeil(input_c, 8);

      auto get_element = [&](int x, int y, int z) -> int8_t {
        auto base = mxc_weights->data<int8_t>() + f * filter_size;
        if (z >= input_c) {
          return 0;
        }
        int idx = filter_h * filter_w * z + filter_w * y + x;
        return base[idx];
      };

      for (int xy = 0; xy < filter_h * filter_w; ++xy) {
        int x = xy % filter_w;
        int y = xy / filter_w;
        for (int z = 0; z < depth; ++z) {
          uint64_t word = 0;
          for (int i = 0; i < 8; ++i) {
            auto elem = get_element(x, y, 8 * z + i);
            weight_sum += elem;
            word |= uint64_t(uint8_t(elem)) << (8 * i);
          }
          fprintf(fp, "%016lX ", word);
        }
      }  // kxy
    }

    // Footer

    auto bias = config.mxc_biases()[f];
    auto scale = config.mxc_scales()[f];

    // Add unsigned input compensation term
    if (idty == x220::DataType::DTY_UINT8) {
      bias.field.bias += weight_sum * 128;
    }
    fprintf(fp, "%016lX %016lX\n", bias.word, scale.word);

  }  // for each filter

  fclose(fp);

  sprintf(path, "./dump/int_weight_%03d.txt", idx_layer);
  fp = fopen(path, "w");

  for (int f = 0; f < output_c; ++f) {
    auto get_element = [&](int x, int y, int z) -> int8_t {
      auto base = mxc_weights->data<int8_t>() + f * filter_size;
      if (z >= input_c) {
        return 0;
      }
      int idx = filter_h * filter_w * z + filter_w * y + x;
      return base[idx];
    };

    // index is not an incremental order due to darknet_mxconv (reference)
    for (int y = 0; y < filter_h; ++y) {
      for (int x = 0; x < filter_w; ++x) {
        for (int c = 0; c < input_c; ++c) {
          auto elem = get_element(x, y, c);
          int index =
              f * filter_size + filter_h * filter_w * c + filter_w * y + x;
          fprintf(fp, "output[%d] = %d\n", index, (int)elem);
        }
      }
    }
  }
  fclose(fp);

  layer.filter(std::move(mxc_weights));
}

void SCOPE::QuantizeLayer(Layer &layer) {}
}  // namespace x220
