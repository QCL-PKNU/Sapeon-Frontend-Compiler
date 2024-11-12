#ifndef QUANTIZATION_X330_QUANT_CONFIG_HPP
#define QUANTIZATION_X330_QUANT_CONFIG_HPP

#include <vector>

#include "datatype.hpp"
#include "x330/aixv_base.h"
#include "x330/aixv_float.h"

namespace x330 {

class QuantConfig {
 public:
  enum class WcalMode { WCAL_NONE = 0, WCAL_LAYER, WCAL_FILTER };
  enum class FcalMode { FCAL_NONE = 0, FCAL_SET, FCAL_ADD, FCAL_MIN };

  dty::DataType input_dtype;
  dty::DataType actin_dtype;
  dty::DataType output_dtype;
  dty::DataType weight_dtype;
  dty::DataType bias_dtype;

  int input_ebias;
  int actin_ebias;
  int output_ebias;
  int weight_ebias;
  int bias_ebias;

  RoundMode input_rmode;
  RoundMode actin_rmode;
  RoundMode output_rmode;
  RoundMode weight_rmode;
  RoundMode bias_rmode;

  FcalMode input_calib;
  FcalMode actin_calib;
  FcalMode output_calib;

  WcalMode weight_calib;

  bool actfn_lut;

  int num_samples = 0;
  double input_max = 0.0;
  double actin_max = 0.0;
  double output_max = 0.0;

  bool dump_debug;  // dump various info
};
}  // namespace x330

#endif  // QUANTIZATION_X330_QUANT_CONFIG_HPP
