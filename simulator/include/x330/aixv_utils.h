#pragma once

#include <cmath>
#include <numeric>

#include "datatype.hpp"
#include "enums/to_underlying_type.hpp"
#include "x330/aixv_base.h"
#include "x330/aixv_float.h"

namespace x330 {

static float GetDtypeMax(dty::DataType dtype, int extra_exp_bias = 0) {
  switch (dtype) {
    case dty::DataType::NF8:
      return NF8 ::Biggest().ToFloat(extra_exp_bias);
    case dty::DataType::NF8U:
      return NF8U ::Biggest().ToFloat(extra_exp_bias);
    case dty::DataType::NF8E:
      return NF8E ::Biggest().ToFloat(extra_exp_bias);
    case dty::DataType::NF9E:
      return NF9E ::Biggest().ToFloat(extra_exp_bias);
    case dty::DataType::NF10:
      return NF10 ::Biggest().ToFloat(extra_exp_bias);
    case dty::DataType::NF10U:
      return NF10U::Biggest().ToFloat(extra_exp_bias);
    case dty::DataType::NF12:
      return NF12 ::Biggest().ToFloat(extra_exp_bias);
    case dty::DataType::NF13:
      return NF13 ::Biggest().ToFloat(extra_exp_bias);
    case dty::DataType::NF13E:
      return NF13E::Biggest().ToFloat(extra_exp_bias);
    case dty::DataType::NF14:
      return NF14 ::Biggest().ToFloat(extra_exp_bias);
    case dty::DataType::NF14E:
      return NF14E::Biggest().ToFloat(extra_exp_bias);
    case dty::DataType::NF15E:
      return NF15E::Biggest().ToFloat(extra_exp_bias);
    case dty::DataType::NF16:
      return NF16 ::Biggest().ToFloat(extra_exp_bias);
    case dty::DataType::BF16:
      return BF16 ::Biggest().ToFloat(extra_exp_bias);
    case dty::DataType::FP32:
      return std::numeric_limits<float>::max();
    default:
      LOG(ERROR) << "invalid data type : "
                 << spgraph_simulator::ToUnderlyingType(dtype);
      exit(1);
  }
}

static void ConvertX330Data(dty::DataType dtype, int ebias, RoundMode rmode,
                            float* data, size_t dim_size) {
  switch (dtype) {
    case dty::DataType::NF8:
#pragma omp parallel for
      for (int i = 0; i < dim_size; ++i) {
        data[i] = NF8::Emulate(data[i], ebias, rmode, true);
      }
      return;
    case dty::DataType::NF8U:
#pragma omp parallel for
      for (int i = 0; i < dim_size; ++i) {
        data[i] = NF8U::Emulate(data[i], ebias, rmode, true);
      }
      return;
    case dty::DataType::NF8E:
#pragma omp parallel for
      for (int i = 0; i < dim_size; ++i) {
        data[i] = NF8E::Emulate(data[i], ebias, rmode, true);
      }
      return;
    case dty::DataType::NF9E:
#pragma omp parallel for
      for (int i = 0; i < dim_size; ++i) {
        data[i] = NF9E::Emulate(data[i], ebias, rmode, true);
      }
      return;
    case dty::DataType::NF10:
#pragma omp parallel for
      for (int i = 0; i < dim_size; ++i) {
        data[i] = NF10::Emulate(data[i], ebias, rmode, true);
      }
      return;
    case dty::DataType::NF10U:
#pragma omp parallel for
      for (int i = 0; i < dim_size; ++i) {
        data[i] = NF10U::Emulate(data[i], ebias, rmode, true);
      }
      return;
    case dty::DataType::NF12:
#pragma omp parallel for
      for (int i = 0; i < dim_size; ++i) {
        data[i] = NF12::Emulate(data[i], ebias, rmode, true);
      }
      return;
    case dty::DataType::NF13:
#pragma omp parallel for
      for (int i = 0; i < dim_size; ++i) {
        data[i] = NF13::Emulate(data[i], ebias, rmode, true);
      }
      return;
    case dty::DataType::NF13E:
#pragma omp parallel for
      for (int i = 0; i < dim_size; ++i) {
        data[i] = NF13E::Emulate(data[i], ebias, rmode, true);
      }
      return;
    case dty::DataType::NF14:
#pragma omp parallel for
      for (int i = 0; i < dim_size; ++i) {
        data[i] = NF14::Emulate(data[i], ebias, rmode, true);
      }
      return;
    case dty::DataType::NF14E:
#pragma omp parallel for
      for (int i = 0; i < dim_size; ++i) {
        data[i] = NF14E::Emulate(data[i], ebias, rmode, true);
      }
      return;
    case dty::DataType::NF15E:
#pragma omp parallel for
      for (int i = 0; i < dim_size; ++i) {
        data[i] = NF15E::Emulate(data[i], ebias, rmode, true);
      }
      return;
    case dty::DataType::NF16:
#pragma omp parallel for
      for (int i = 0; i < dim_size; ++i) {
        data[i] = NF16::Emulate(data[i], ebias, rmode, true);
      }
      return;
    case dty::DataType::BF16:
#pragma omp parallel for
      for (int i = 0; i < dim_size; ++i) {
        data[i] = BF16::Emulate(data[i], ebias, rmode, true);
      }
      return;
    case dty::DataType::FP32:
      return;
    default:
      LOG(ERROR) << "invalid data type : "
                 << spgraph_simulator::ToUnderlyingType(dtype);
      exit(1);
  }
}

static float CalculateX330Max(const float* const data, const size_t dim_size) {
  float max_value = std::numeric_limits<float>::min();
#pragma omp parallel for reduction(max : max_value)
  for (int i = 0; i < dim_size; ++i) {
    const float abs_value = std::abs(data[i]);
    if (std::isfinite(abs_value) && abs_value != 0.0f) {
      max_value = std::max(max_value, abs_value);
    }
  }
  return max_value;
}
}  // namespace x330
