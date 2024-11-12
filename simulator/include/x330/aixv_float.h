//==============================================================================
// (C) 2023 SAPEON Korea Inc. All rights reserved.
//==============================================================================

#pragma once

#include <cstdint>

namespace x330 {

enum class RoundMode {
  ROUND_TO_NEAREST_EVEN = 0,
  ROUND_TO_NEAREST_UP,
  ROUND_TO_ZERO
};

template <typename T, int S, int E, int M, bool COMPACT>
class NeuralFloat {
 public:
  NeuralFloat() : bits_(0) {}

  //----------------------------------------------------------------------------
  // Static methods
  //----------------------------------------------------------------------------

  static int mantissa_field_bits() { return M; }
  static int base_exp_bias() { return (1 << E) / 2 - 1; }
  static int min_normal_encoding_exp() { return COMPACT ? 0 : 1; }
  static int max_normal_encoding_exp() {
    return (1 << E) - ((COMPACT && S) ? 1 : 2);
  }

  static NeuralFloat<T, S, E, M, COMPACT> FromBits(T bits) {
    NeuralFloat<T, S, E, M, COMPACT> nf;
    nf.bits_ = bits;
    return nf;
  }

  static NeuralFloat<T, S, E, M, COMPACT> FromFields(int s, int e, int m) {
    T bits = ((s & S) << (E + M)) | ((e & ((1 << E) - 1)) << M) |
             ((m & ((1 << M) - 1)) << 0);
    return FromBits(bits);
  }

  static NeuralFloat<T, S, E, M, COMPACT> Zero(int s = 0) {
    constexpr bool kNoNegZero =
        true;  // Negative zero causes complex situations...
    if (COMPACT || kNoNegZero) {
      return FromFields(0, 0, 0);
    } else {
      return FromFields(s, 0, 0);
    }
  }

  static NeuralFloat<T, S, E, M, COMPACT> Biggest(int s = 0) {
    return FromFields(s, max_normal_encoding_exp(), (1 << M) - 1);
  }

  static NeuralFloat<T, S, E, M, COMPACT> Infinity(int s = 0) {
    if (COMPACT && S) {
      return FromFields(1, 0, 0);
    } else {
      return FromFields(s, (1 << E) - 1, 0);
    }
  }

  static NeuralFloat<T, S, E, M, COMPACT> FromFloat(
      float fp32, int extra_exp_bias = 0,
      RoundMode rounding_mode = RoundMode::ROUND_TO_NEAREST_EVEN,
      bool saturate = true) {
    union {
      float f;
      U32 u;
    } tmp = {fp32};
    U32 fp32_bits = tmp.u;
    int fp32_s = fp32_bits >> 31;
    int fp32_e = ((fp32_bits >> 23) & 0xFF) - 127;
    int fp32_m = fp32_bits & 0x7FFFFF;
    if (fp32_e == -127) {
      return Zero(fp32_s);
    }
    if (fp32_e == 128) {
      return Infinity(fp32_s);
    }
    if (fp32_s && S == 0) {
      return Zero();
    }

    // LF   mantissa ..............
    // FP32 mantissa .............|||~~~~~~~~|
    //                           / | \      /
    //                          L  R  Sticky
    int l_bit = (fp32_m >> (23 - M)) & 0x1;
    int r_bit = (fp32_m >> (23 - M - 1)) & 0x1;
    int sticky = fp32_m & ((1 << (23 - M - 1)) - 1);

    int mantissa = fp32_m >> (23 - M);
    int exponent = fp32_e + base_exp_bias() + extra_exp_bias;
    if (rounding_mode == RoundMode::ROUND_TO_NEAREST_EVEN) {
      mantissa += r_bit & (l_bit | (sticky != 0 ? 1 : 0));
    } else if (rounding_mode == RoundMode::ROUND_TO_NEAREST_UP) {
      mantissa += r_bit;
    }

    exponent += mantissa >> M;
    mantissa &= (1 << M) - 1;

    int exp_min = min_normal_encoding_exp();
    int exp_max = max_normal_encoding_exp();

    if (exponent < exp_min) return Zero(fp32_s);
    if (exponent > exp_max) {
      if (saturate) {
        return Biggest(fp32_s);
      } else {
        return Infinity(fp32_s);
      }
    }

    if (COMPACT) {
      if (exponent == 0 && mantissa == 0) return Zero();
    }

    return FromFields(fp32_s, exponent, mantissa);
  }

  static NeuralFloat<T, S, E, M, COMPACT> FromInt64(
      S64 i64, int exp_offset = 0,
      RoundMode rounding_mode = RoundMode::ROUND_TO_NEAREST_EVEN,
      bool saturate = true) {
    if (i64 == 0) return Zero();
    if (i64 < 0 && S == 0) return Zero();

    int sign = i64 < 0 ? 1 : 0;
    U64 magnitude = sign ? -i64 : i64;
    int clz = CountLeadingZeroBits(magnitude);
    magnitude <<= clz + 1;

    int l_bit = (magnitude >> (64 - M)) & 0x1;
    int r_bit = (magnitude >> (64 - M - 1)) & 0x1;
    U64 sticky = magnitude & ((1ULL << (64 - M - 1)) - 1);

    int mantissa = magnitude >> (64 - M);
    int exponent = (63 - clz) + base_exp_bias() + exp_offset;

    if (rounding_mode == RoundMode::ROUND_TO_NEAREST_EVEN) {
      mantissa += r_bit & (l_bit | (sticky != 0 ? 1 : 0));
    } else if (rounding_mode == RoundMode::ROUND_TO_NEAREST_UP) {
      mantissa += r_bit;
    }

    exponent += mantissa >> M;
    mantissa &= (1 << M) - 1;

    int exp_min = min_normal_encoding_exp();
    int exp_max = max_normal_encoding_exp();

    if (exponent < exp_min) return Zero(sign);
    if (exponent > exp_max) {
      if (saturate) {
        return Biggest(sign);
      } else {
        return Infinity(sign);
      }
    }

    if (COMPACT) {
      if (exponent == 0 && mantissa == 0) return Zero();
    }

    return FromFields(sign, exponent, mantissa);
  }

  static float Emulate(
      float fp32, int extra_exp_bias,
      RoundMode rounding_mode = RoundMode::ROUND_TO_NEAREST_EVEN,
      bool saturate = true) {
    return FromFloat(fp32, extra_exp_bias, rounding_mode, saturate)
        .ToFloat(extra_exp_bias);
  }

  static NeuralFloat<T, S, E, M, COMPACT> Add(
      NeuralFloat<T, S, E, M, COMPACT> src0,
      NeuralFloat<T, S, E, M, COMPACT> src1,
      RoundMode rounding_mode = RoundMode::ROUND_TO_NEAREST_EVEN,
      bool saturate = true) {
    int s0_s = src0.GetSignField();
    int s1_s = src1.GetSignField();
    int s0_m = src0.GetMantissaField();
    int s1_m = src1.GetMantissaField();
    int s0_e = src0.GetExponentField();
    int s1_e = src1.GetExponentField();

    if (src0.IsInfinity() || src1.IsInfinity()) {
      return Infinity();
    }

    if (src0.IsZero()) return src1;
    if (src1.IsZero()) return src0;

    if (s0_e < s1_e || (s0_e == s1_e && s0_m < s1_m)) {
      std::swap(s0_s, s1_s);
      std::swap(s0_e, s1_e);
      std::swap(s0_m, s1_m);
    }

    s0_m |= (1 << M);
    s1_m |= (1 << M);
    s0_m = s0_s ? -s0_m : s0_m;
    s1_m = s1_s ? -s1_m : s1_m;

    int exp_diff = std::min(s0_e - s1_e, M + 3);
    int exp_offset = s0_e - (base_exp_bias() + M + exp_diff);

    S64 int_result = (S64(s0_m) << exp_diff) + s1_m;
    return FromInt64(int_result, exp_offset, rounding_mode, saturate);
  }

  static NeuralFloat<T, S, E, M, COMPACT> Mul(
      NeuralFloat<T, S, E, M, COMPACT> src0,
      NeuralFloat<T, S, E, M, COMPACT> src1,
      RoundMode rounding_mode = RoundMode::ROUND_TO_NEAREST_EVEN,
      bool saturate = true) {
    int s0_s = src0.GetSignField();
    int s1_s = src1.GetSignField();
    int s0_m = src0.GetMantissaField();
    int s1_m = src1.GetMantissaField();
    int s0_e = src0.GetExponentField();
    int s1_e = src1.GetExponentField();

    if (src0.IsInfinity() || src1.IsInfinity()) {
      return Infinity((s0_s ^ s1_s) & 0x1);
    }
    if (src0.IsZero() || src1.IsZero()) {
      return Zero((s0_s ^ s1_s) & 0x1);
    }

    s0_m |= (1 << M);
    s1_m |= (1 << M);
    s0_m = s0_s ? -s0_m : s0_m;
    s1_m = s1_s ? -s1_m : s1_m;

    int exp_offset = s0_e + s1_e - 2 * (base_exp_bias() + M);

    S64 int_result = S64(s0_m) * s1_m;
    return FromInt64(int_result, exp_offset, rounding_mode, saturate);
  }

  static NeuralFloat<T, S, E, M, COMPACT> Fma(
      NeuralFloat<T, S, E, M, COMPACT> src0,
      NeuralFloat<T, S, E, M, COMPACT> src1,
      NeuralFloat<T, S, E, M, COMPACT> src2,
      RoundMode rounding_mode = RoundMode::ROUND_TO_NEAREST_EVEN,
      bool saturate = true) {
    int s0_s = src0.GetSignField();
    int s1_s = src1.GetSignField();
    int s2_s = src2.GetSignField();
    int s0_m = src0.GetMantissaField() | (1 << M);
    int s1_m = src1.GetMantissaField() | (1 << M);
    int s2_m = src2.GetMantissaField() | (1 << M);
    int s0_e = src0.GetExponentField();
    int s1_e = src1.GetExponentField();
    int s2_e = src2.GetExponentField();

    if (src0.IsInfinity() || src1.IsInfinity() || src2.IsInfinity()) {
      return Infinity();
    }

    if (src0.IsZero() || src1.IsZero()) {
      s0_m = 0;
      s0_e = 0;
      s1_m = 0;
      s1_e = 0;
    }
    if (src2.IsZero()) {
      s2_m = 0;
      s2_e = 0;
    }
    s0_m = s0_s ? -s0_m : s0_m;
    s1_m = s1_s ? -s1_m : s1_m;
    s2_m = s2_s ? -s2_m : s2_m;

    S64 int_product = S64(s0_m) * s1_m;
    int s0_exp_offset = s0_e - (base_exp_bias() + M);
    int s1_exp_offset = s1_e - (base_exp_bias() + M);
    int s2_exp_offset = s2_e - (base_exp_bias() + M);
    int exp_diff = s0_exp_offset + s1_exp_offset - s2_exp_offset;

    int exp_offset;
    S64 int_sum;
    if (exp_diff >= 0) {
      exp_diff = std::min(exp_diff, M + 1);
      exp_offset = s0_exp_offset + s1_exp_offset - exp_diff;
      int_sum = (int_product << exp_diff) + s2_m;
    } else {
      exp_diff = std::min(-exp_diff, 2 * M + 3);
      exp_offset = s2_exp_offset - exp_diff;
      int_sum = (S64(s2_m) << exp_diff) + int_product;
    }
    return FromInt64(int_sum, exp_offset, rounding_mode, saturate);
  }

  static NeuralFloat<T, S, E, M, COMPACT> Min(
      NeuralFloat<T, S, E, M, COMPACT> src0,
      NeuralFloat<T, S, E, M, COMPACT> src1) {
    int s0_s = src0.GetSignField();
    int s1_s = src1.GetSignField();
    int s0_m = src0.GetMantissaField();
    int s1_m = src1.GetMantissaField();
    int s0_e = src0.GetExponentField();
    int s1_e = src1.GetExponentField();

    if (src0.IsInfinity() || src1.IsInfinity()) {
      return Infinity();
    }

    int s0_em = (s0_e << M) | s0_m;
    int s1_em = (s1_e << M) | s1_m;
    int lt_em = s0_em < s1_em;
    if (s0_s < s1_s) return src1;  // + vs -
    if (s0_s > s1_s) return src0;  // - vs +
    if (s0_s)
      return lt_em ? src1 : src0;
    else
      return lt_em ? src0 : src1;
  }

  static NeuralFloat<T, S, E, M, COMPACT> Max(
      NeuralFloat<T, S, E, M, COMPACT> src0,
      NeuralFloat<T, S, E, M, COMPACT> src1) {
    int s0_s = src0.GetSignField();
    int s1_s = src1.GetSignField();
    int s0_m = src0.GetMantissaField();
    int s1_m = src1.GetMantissaField();
    int s0_e = src0.GetExponentField();
    int s1_e = src1.GetExponentField();

    if (src0.IsInfinity() || src1.IsInfinity()) {
      return Infinity();
    }

    int s0_em = (s0_e << M) | s0_m;
    int s1_em = (s1_e << M) | s1_m;
    int lt_em = s0_em < s1_em;
    if (s0_s < s1_s) return src0;  // + vs -
    if (s0_s > s1_s) return src1;  // - vs +
    if (s0_s)
      return lt_em ? src0 : src1;
    else
      return lt_em ? src1 : src0;
  }

  //----------------------------------------------------------------------------
  // Instance Methods
  //----------------------------------------------------------------------------
  int GetSignField() const { return (bits_ >> (E + M)) & 0x1; }
  int GetExponentField() const { return (bits_ >> M) & ((1 << E) - 1); }
  int GetMantissaField() const { return bits_ & ((1 << M) - 1); }

  int GetSignedSignificand() const {
    if (IsZero()) return 0;
    int s = GetSignField();
    int m = (1 << M) | GetMantissaField();
    return s ? -m : m;
  }

  bool IsNegative() const { return GetSignField() == 1; }
  bool IsPositive() const { return GetSignField() == 0; }

  bool IsZero() const {
    if (COMPACT) {
      return GetSignField() == 0 && GetExponentField() == 0 &&
             GetMantissaField() == 0;
    } else {
      return GetExponentField() == 0;
    }
  }

  bool IsInfinity() const {
    if (COMPACT && S) {
      return GetSignField() == 1 && GetExponentField() == 0 &&
             GetMantissaField() == 0;
    } else {
      return GetExponentField() == ((1 << E) - 1);
    }
  }

  T ToBits() const { return bits_; }

  float ToFloat(int extra_exp_bias = 0) const {
    int fp32_s = GetSignField();
    int fp32_e;
    int fp32_m;

    if (IsZero()) {
      fp32_e = 0;
      fp32_m = 0;
    } else if (IsInfinity()) {
      fp32_s &= COMPACT ? 0 : 1;
      fp32_e = 0xFF;
      fp32_m = 0;
    } else {
      fp32_m = GetMantissaField() << (23 - M);
      fp32_e = GetExponentField() + 127 - base_exp_bias() - extra_exp_bias;
    }

    U32 fp32_bits = (fp32_s << 31) | (fp32_e << 23) | (fp32_m);
    union {
      U32 u;
      float f;
    } tmp = {fp32_bits};
    return tmp.f;
  }

  S64 ToInt64(
      int extra_exp_bias = 0,
      RoundMode rounding_mode = RoundMode::ROUND_TO_NEAREST_EVEN) const {
    if (IsInfinity()) {
      return (1ULL << 63) - 1;
    }
    if (IsZero()) {
      return 0;
    }

    int s = GetSignField();
    S64 m = GetMantissaField() | (1 << M);
    int e = GetExponentField() - base_exp_bias() - extra_exp_bias - M;

    if (e >= 0) {
      if (e >= 64 - M - 2) {
        return (1ULL << 63) - (s > 0 ? 1 : 0);
      }
      m <<= e;
      return s ? -m : m;
    }

    m <<= 2;
    e = std::min(-e, M + 2);

    int l_bit = (m >> (e + 2)) & 0x1;
    int r_bit = (m >> (e + 1)) & 0x1;
    int sticky = m & ((1ULL << (e + 1)) - 1);

    m >>= e + 2;
    if (rounding_mode == RoundMode::ROUND_TO_NEAREST_EVEN) {
      m += r_bit & (l_bit | (sticky != 0 ? 1 : 0));
    } else if (rounding_mode == RoundMode::ROUND_TO_NEAREST_UP) {
      m += r_bit;
    }

    return s ? -m : m;
  }

 private:
  T bits_;
};

using U8 = uint8_t;
using U16 = uint16_t;
using NF8 = NeuralFloat<U8, 1, 4, 3, true>;
using NF8U = NeuralFloat<U8, 0, 4, 4, true>;
using NF8E = NeuralFloat<U8, 1, 5, 2, true>;
using NF10 = NeuralFloat<U16, 1, 4, 5, true>;
using NF10U = NeuralFloat<U16, 0, 4, 6, true>;
using NF16 = NeuralFloat<U16, 1, 6, 9, false>;
using BF16 = NeuralFloat<U16, 1, 8, 7, false>;
// Intermediate types
using NF7 = NeuralFloat<U8, 1, 4, 2, true>;
using NF9E = NeuralFloat<U16, 1, 5, 3, true>;
using NF12 = NeuralFloat<U16, 1, 4, 7, true>;
using NF13 = NeuralFloat<U16, 1, 4, 8, true>;
using NF13E = NeuralFloat<U16, 1, 5, 7, true>;
using NF14 = NeuralFloat<U16, 1, 4, 9, true>;
using NF14E = NeuralFloat<U16, 1, 5, 8, true>;
using NF15E = NeuralFloat<U16, 1, 5, 9, true>;

}  // namespace x330
