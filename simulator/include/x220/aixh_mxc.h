#ifndef MXC_AIXH_MXC_H
#define MXC_AIXH_MXC_H

#include <math.h>

#include <algorithm>

#include "x220/aixh_common.h"

namespace x220 {

// Filter bias format
union MxcBias {
  U64 word;
  struct {
    U64 bias : 48;
    U64 __pad__ : 16;
  } field;

  MxcBias(U64 w = 0) { word = w; }

  void Encode(S64 acc_offset, float weight_scale, float in_scale, float bias) {
    this->field.bias = acc_offset + S64(round(bias * in_scale * weight_scale));
  }

  /// encode fixed bias after batch normalization & scale
  /// G' = (G-M)/(sqrt(var)+e)*scale+bias = G/alpha+bias-mean/alpha
  ///    = G/alpha(=(sqrt(var)+e)*scale) + new_bias(=bias-mean/alpha)
  void Encode(S64 acc_offset, float weight_scale, float in_scale, float alpha,
              float mean, float bias) {
    float new_bias = bias - mean / alpha;
    this->field.bias =
        acc_offset + S64(round(new_bias * in_scale * weight_scale));
  }
};

// Filter scale format
union MxcScale {
  U64 word;
  struct {
    U64 pos_scale_mantissa : MXC_SCALE_MANTISSA_BITS - 1;
    U64 pos_scale_exponent : 6;
    U64 neg_scale_mantissa : MXC_SCALE_MANTISSA_BITS - 1;
    U64 neg_scale_exponent : 6;
  } field;

  MxcScale(U64 w = 0) { word = w; }

  static float __GetScaleFp32(int mantissa, int exponent) {
    const int kSmBits = MXC_SCALE_MANTISSA_BITS;  // Scale Mantissa
    const int kInBits = 48;                       // Input Bits
    U32 res_u32;
    if (exponent == 0 && mantissa == 0) return 0.0f;
    exponent += 127;  // Fp32 bias
    exponent -= kInBits;

    res_u32 = (mantissa << (24 - kSmBits)) | (exponent << 23);
    return BitcastU32ToF32(res_u32);
  }

  float GetPosScaleFp32() {
    return __GetScaleFp32(this->field.pos_scale_mantissa,
                          this->field.pos_scale_exponent);
  }

  float GetNegScaleFp32() {
    return __GetScaleFp32(this->field.neg_scale_mantissa,
                          this->field.neg_scale_exponent);
  }

  static void __EncodeScale(float scale_fp32, int& mantissa, int& exponent) {
    const int kSmBits = MXC_SCALE_MANTISSA_BITS;  // Scale Mantissa
    const int kSeBits = 6;                        // Scale Exponent Bits
    const int kInBits = 48;                       // Input Bits
    U32 scale_u32 = BitcastF32ToU32(scale_fp32);
    mantissa = (scale_u32 >> 0) & 0x7FFFFF;
    exponent = (scale_u32 >> 23) & 0xFF;
    // Convert to '1.manttisa * 2^kInputBits * 2^exponent' form
    exponent -= 127;
    exponent += kInBits;

    // Round mantissa and shift
    const int kSmTrunc = 24 - kSmBits;
    const int kStickyMask = (1 << (kSmTrunc - 1)) - 1;
    int sticky_bit = (mantissa & kStickyMask) != 0 ? 1 : 0;
    int round_bit = (mantissa >> (kSmTrunc - 1)) & 0x1;
    int guide_bit = (mantissa >> (kSmTrunc - 0)) & 0x1;
    mantissa >>= kSmTrunc;
    if (round_bit & (sticky_bit | guide_bit)) {
      mantissa++;
    }
    // Overflow handling
    if ((mantissa >> (kSmBits - 1)) != 0) {
      mantissa = 0;
      exponent++;
    }

    const int kSeMax = (1 << kSeBits) - 1;
    exponent = std::max(0, std::min(exponent, kSeMax));

    // Special handling for zero
    if (scale_fp32 == 0.0f) {
      mantissa = 0;
      exponent = 0;
    }
  }

  // Adjust FP32 to the equivalent encoded scale.
  static float AdjustFp32(float raw_fp32) {
    int mantissa;
    int exponent;
    __EncodeScale(raw_fp32, mantissa, exponent);
    return __GetScaleFp32(mantissa, exponent);
  }

  // Adjust weight scale
  static float AdjustWeightScale(float w_scale, float in_scale,
                                 float out_scale) {
    float scale = AdjustFp32(out_scale / (in_scale * w_scale));
    return out_scale / (scale * in_scale);
  }

  // Adjust output scale
  static float AdjustOutputScale(float w_scale, float in_scale,
                                 float out_scale) {
    float scale = AdjustFp32(out_scale / (in_scale * w_scale));
    return scale * (in_scale * w_scale);
  }

  // Encode the scale fields
  void Encode(float weight_scale, float in_scale, float out_scale,
              float neg_slope) {
    float pos_scale = out_scale / (in_scale * weight_scale);
    float neg_scale = pos_scale * neg_slope;

    int pos_scale_mantissa;
    int neg_scale_mantissa;
    int pos_scale_exponent;
    int neg_scale_exponent;
    __EncodeScale(pos_scale, pos_scale_mantissa, pos_scale_exponent);
    __EncodeScale(neg_scale, neg_scale_mantissa, neg_scale_exponent);

    this->field.pos_scale_mantissa = pos_scale_mantissa;
    this->field.neg_scale_mantissa = neg_scale_mantissa;
    this->field.pos_scale_exponent = pos_scale_exponent;
    this->field.neg_scale_exponent = neg_scale_exponent;
  }

  // Scale an input
  int Scale(S64 in, DataType out_dtype) {
    const int kSmBits = MXC_SCALE_MANTISSA_BITS;  // Scale Mantissa
    const int kInBits = 48;                       // Input Bits
    const int kImBits = 24;                       // Input Mantissa Bits
    const int kIeMax = kInBits - 1 - kImBits;

    //
    // Convert the input in an FP32-equivalent form
    //
    in = SignExtend(in, kInBits);
    bool negative = in < 0;
    if (negative) in = -in;

    // LZD, normalize and round
    int in_leading0s = CountLeadingZeroBits(in, kInBits - 1);
    int in_exponent = std::max(0, kIeMax - in_leading0s);
    U64 in_point5 = 1ULL << in_exponent >> 1;
    U64 in_mantissa = (in + in_point5) >> in_exponent;
    if ((in_mantissa >> kImBits) != 0) {
      in_mantissa = 1ULL << (kImBits - 1);
      in_exponent++;
    }

    //
    // Multiply
    //
    U64 scale_mantissa;
    int scale_exponent;
    if (negative) {
      scale_mantissa = this->field.neg_scale_mantissa;
      scale_exponent = this->field.neg_scale_exponent;
    } else {
      scale_mantissa = this->field.pos_scale_mantissa;
      scale_exponent = this->field.pos_scale_exponent;
    }
    scale_mantissa |= 1ULL << (kSmBits - 1);  // hidden bit

    U64 out_mantissa = in_mantissa * scale_mantissa;
    int out_exponent = in_exponent + scale_exponent - (kInBits + kSmBits - 1);

    const int kPreLshamt = 16;
    int out_rshamt = std::max(0, std::min(kPreLshamt - 1 - out_exponent, 63));
    U64 out_rshifted = out_mantissa << kPreLshamt >> out_rshamt;
    U64 out_rounded = (out_rshifted + 1) >> 1;

    //
    // Convert back to the fixed-point
    //
    int out_mag_bits = out_dtype == DataType::DTY_SINT16  ? 15
                       : out_dtype == DataType::DTY_UINT8 ? 8
                       : out_dtype == DataType::DTY_SINT8 ? 7
                       : out_dtype == DataType::DTY_UINT4 ? 4
                                                          : 3;

    bool overflow = out_rounded >> out_mag_bits != 0;
    int out_saturated = out_rounded;
    if (overflow) {
      out_saturated = (1 << out_mag_bits) - 1;
    }
    if (negative && IsUnsigned(out_dtype)) {
      out_saturated = 0;
    }

    int out_result = negative ? -out_saturated : out_saturated;
    return out_result;
  }
};

}  // end of namespace x220

#endif  // MXC_AIXH_MXC_H
