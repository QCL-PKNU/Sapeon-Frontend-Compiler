#ifndef QUANTIZATION_X220_QUANT_CONFIG_HPP
#define QUANTIZATION_X220_QUANT_CONFIG_HPP

#include <vector>

#include "x220/aixh_mxc.h"

namespace x220 {

// TODO: remove unnecessary variables and rename

class QuantConfig {
 public:
  QuantConfig() {}
  virtual ~QuantConfig() {}

  struct Shortcut {
    int a_mul;
    int b_mul;
    int rsh;
    int64_t bias;
  };

  struct Asqnr {
    int nsamples;
    float accumulated;
  };  // activation SQNR

  struct Astat {
    int nsamples;
    double quantiles[3 * 5];
    double real_in_th[3 * 5];
  };  // activation statistics

  int wquant_en() const { return wquant_en_; }
  void wquant_en(int value) { wquant_en_ = value; }

  int aquant_en() const { return aquant_en_; }
  void aquant_en(int value) { aquant_en_ = value; }

  DataType out_dtype() const { return out_dtype_; }
  void out_dtype(DataType value) { out_dtype_ = value; }

  DataType in_dtype() const { return in_dtype_; }
  void in_dtype(DataType value) { in_dtype_ = value; }

  int wamplifier() const { return wamplifier_; }
  void wamplifier(int value) { wamplifier_ = value; }

  int calib_mode() const { return calib_mode_; }
  void calib_mode(int value) { calib_mode_ = value; }

  float calib_adj() const { return calib_adj_; }
  void calib_adj(float value) { calib_adj_ = value; }

  int multiplier() const { return multiplier_; }
  void multiplier(int value) { multiplier_ = value; }

  int shifter() const { return shifter_; }
  void shifter(int value) { shifter_ = value; }

  float iscale() const { return iscale_; }
  void iscale(float value) { iscale_ = value; }

  float oscale() const { return oscale_; }
  void oscale(float value) { oscale_ = value; }

  float oqbias() const { return oqbias_; }
  void oqbias(float value) { oqbias_ = value; }

  float oqmax() const { return oqmax_; }
  void oqmax(float value) { oqmax_ = value; }

  float oqmin() const { return oqmin_; }
  void oqmin(float value) { oqmin_ = value; }

  const Asqnr& asqnr() const { return asqnr_; }
  void asqnr(Asqnr value) { asqnr_ = value; }

  const Astat& astat() const { return astat_; }
  void astat(Astat value) { astat_ = value; }

  const Shortcut& shortcut() const { return shortcut_; }
  void shortcut(Shortcut value) { shortcut_ = value; }

  const std::vector<float>& weights() const { return weights_; }
  void weights(const std::vector<float>& values) { weights_ = values; }

  const std::vector<float>& biases() const { return biases_; }
  void biases(const std::vector<float>& values) { biases_ = values; }

  const std::vector<float>& workspace() const { return workspace_; }
  void workspace(const std::vector<float>& values) { workspace_ = values; }

  const std::vector<MxcBias>& mxc_biases() const { return mxc_biases_; }
  void mxc_biases(const std::vector<MxcBias>& values) { mxc_biases_ = values; }

  const std::vector<MxcScale>& mxc_scales() const { return mxc_scales_; }
  void mxc_scales(const std::vector<MxcScale>& values) { mxc_scales_ = values; }

 private:
  int strict_mode_;  // strict HW modeling mode
  int approx_int8_;  // approximate INT8 (NI8 for X330)
  int dump_astat_;   // dump activation statistics
  int dump_asqnr_;   // dump activation SQNR
  int dump_data_;    // dump data including weights
  int dump_debug_;   // dump various info

  int quant_en_;        // quantization enabled
  int wquant_en_;       // weight quantization enabled
  int aquant_en_;       // activation quantization enabled
  int out_dequant_;     // output dequantization enabled
  DataType in_dtype_;   // input data type
  DataType out_dtype_;  // output data type
  int wamplifier_;      // weight amplification factor
  int calib_mode_;      // calibration mode
  float calib_adj_;     // calibration scale

  float iscale_;  // input scale
  float oscale_;  // output scale
  float oqbias_;  // output quantized bias
  float oqmin_;   // output quantized min
  float oqmax_;   // output quantized max
  int oquant_;    // output quantized or not

  std::vector<float> weights_;
  std::vector<float> biases_;
  std::vector<float> workspace_;

  Asqnr asqnr_;
  Astat astat_;

  Shortcut shortcut_;

  // NVP parameter
  int multiplier_;
  int shifter_;

  std::vector<MxcBias> mxc_biases_;
  std::vector<MxcScale> mxc_scales_;
};
}  // namespace x220

#endif  // QUANTIZATION_X220_QUANT_CONFIG_HPP
