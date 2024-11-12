//==============================================================================
// (C) 2023 SAPEON Korea Inc. All rights reserved.
//==============================================================================

#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "datatype.hpp"
#include "glog/logging.h"

namespace x330 {
/*----------------------------------------------------------------------------
 * Primitive Types
 *----------------------------------------------------------------------------*/
// Integer types - legacy naming
using S8 = int8_t;
using S16 = int16_t;
using S32 = int32_t;
using S64 = int64_t;
using U8 = uint8_t;
using U16 = uint16_t;
using U32 = uint32_t;
using U64 = uint64_t;

// Integer types - alternative naming
using I8 = int8_t;
using I16 = int16_t;
using I32 = int32_t;
using I64 = int64_t;
using I8U = uint8_t;
using I16U = uint16_t;
using I32U = uint32_t;
using I64U = uint64_t;

// Standard floating-point types
using FP32 = float;
using FP64 = double;

// Tensor cache block word type
using TcbWord = uint64_t;

// Clock cycle counting type
using Cycle = int64_t;

/*----------------------------------------------------------------------------
 * Fused block rate
 *----------------------------------------------------------------------------*/
enum class FusedBlockRate {
  FBR_X1 = 0,
  FBR_X2 = 1,
  FBR_X4 = 2,
  FBR_X8 = 3,
  FBR_X16 = 4
};

/*----------------------------------------------------------------------------
 * Simple tensor class (up to 5-dimensions)
 *----------------------------------------------------------------------------*/
// template <typename T>
// class Tensor {
//  public:
//   Tensor(int s0, int s1 = 1, int s2 = 1, int s3 = 1, int s4 = 1)
//       : shape_{s0, s1, s2, s3, s4} {
//     stride_[3] = s4;
//     stride_[2] = s4 * s3;
//     stride_[1] = s4 * s3 * s2;
//     stride_[0] = s4 * s3 * s2 * s1;
//     mem_.resize(s4 * s3 * s2 * s1 * s0);
//   }

//   T& operator()(int d0, int d1 = 0, int d2 = 0, int d3 = 0, int d4 = 0) {
//     return mem_[GetIdx(d0, d1, d2, d3, d4)];
//   }
//   std::vector<T>& flat_vector() { return mem_; }
//   T* flat_array() { return mem_.data(); }

//   const T& operator()(int d0, int d1 = 0, int d2 = 0, int d3 = 0,
//                       int d4 = 0) const {
//     return mem_[GetIdx(d0, d1, d2, d3, d4)];
//   }
//   const std::vector<T>& flat_vector() const { return mem_; }
//   const T* flat_array() const { return mem_.data(); }

//   int shape(int dim) const { return shape_[dim]; }
//   int stride(int dim) const { return stride_[dim]; }
//   int size() const { return mem_.size(); }
//   int byte_size() const { return size() * sizeof(T); }

//  private:
//   int GetIdx(int d0, int d1, int d2, int d3, int d4) {
//     return d0 * stride_[0] + d1 * stride_[1] + d2 * stride_[2] +
//            d3 * stride_[3] + d4;
//   }

//  private:
//   std::vector<T> mem_;
//   std::array<int, 5> shape_;
//   std::array<int, 4> stride_;
// };

/*----------------------------------------------------------------------------
 * Utilities
 *----------------------------------------------------------------------------*/
#define AIXV_PREDICT_FALSE(x) __builtin_expect(x, 0)

static inline int IDivCeil(int x, int y) { return (x + y - 1) / y; }

static inline int ICeil(int x, int y) { return IDivCeil(x, y) * y; }

static inline int CountRedundantSignBits(S64 x, int bit_width = 64) {
  int res = __builtin_clrsbll(x) - (64 - bit_width);
  return res;
}
static inline int CountLeadingZeroBits(S64 x, int bit_width = 64) {
  if (x == 0) return bit_width;
  int res = __builtin_clzll(x) - (64 - bit_width);
  return res;
}

static inline constexpr S64 SignExtend(S64 x, int bit_width = 64) {
  x <<= 64 - bit_width;
  x >>= 64 - bit_width;
  return x;
}

static inline U32 BitcastF32ToU32(float x) {
  union {
    float f;
    U32 u;
  } tmp;
  tmp.f = x;
  return tmp.u;
}

static inline float BitcastU32ToF32(U32 x) {
  union {
    float f;
    U32 u;
  } tmp;
  tmp.u = x;
  return tmp.f;
}

static inline constexpr int GetDataTypeExponentBitwidth(dty::DataType dty) {
  switch (dty) {
    case dty::DataType::I8:
      return 0;
    case dty::DataType::I8U:
      return 0;
    case dty::DataType::I10U:
      return 0;
    case dty::DataType::NF8:
      return 4;
    case dty::DataType::NF8U:
      return 4;
    case dty::DataType::NF8E:
      return 5;
    case dty::DataType::NF10:
      return 4;
    case dty::DataType::NF10U:
      return 4;
    case dty::DataType::NF16:
      return 6;
    case dty::DataType::BF16:
      return 8;
    case dty::DataType::FP32:
      return 8;
    default:
      return -1;
  }
}

static inline constexpr int sizeof_dtype(dty::DataType dty,
                                         bool ceiling = true) {
  switch (dty) {
    case dty::DataType::I8:
      return 1;
    case dty::DataType::I8U:
      return 1;
    case dty::DataType::I10U:
      return ceiling ? 2 : 1;
    case dty::DataType::NF8:
      return 1;
    case dty::DataType::NF8U:
      return 1;
    case dty::DataType::NF8E:
      return 1;
    case dty::DataType::NF10:
      return ceiling ? 2 : 1;
    case dty::DataType::NF10U:
      return ceiling ? 2 : 1;
    case dty::DataType::NF16:
      return 2;
    case dty::DataType::BF16:
      return 2;
    case dty::DataType::FP32:
      return 4;
    default:
      return -1;
  }
}

static inline constexpr int bit_sizeof_dtype(dty::DataType dty) {
  switch (dty) {
    case dty::DataType::I8:
      return 8;
    case dty::DataType::I8U:
      return 8;
    case dty::DataType::I10U:
      return 10;
    case dty::DataType::NF8:
      return 8;
    case dty::DataType::NF8U:
      return 8;
    case dty::DataType::NF8E:
      return 8;
    case dty::DataType::NF10:
      return 10;
    case dty::DataType::NF10U:
      return 10;
    case dty::DataType::NF16:
      return 16;
    case dty::DataType::BF16:
      return 16;
    case dty::DataType::FP32:
      return 32;
    default:
      return -1;
  }
}

static inline constexpr const char* nameof_dtype(dty::DataType dty) {
  switch (dty) {
    case dty::DataType::I8:
      return "I8";
    case dty::DataType::I8U:
      return "I8U";
    case dty::DataType::I10U:
      return "I10U";
    case dty::DataType::NF8:
      return "NF8";
    case dty::DataType::NF8U:
      return "NF8U";
    case dty::DataType::NF8E:
      return "NF8E";
    case dty::DataType::NF10:
      return "NF10";
    case dty::DataType::NF10U:
      return "NF10U";
    case dty::DataType::NF16:
      return "NF16";
    case dty::DataType::BF16:
      return "BF16";
    case dty::DataType::FP32:
      return "FP32";
    default:
      return "INVALID";
  }
}

static inline constexpr FusedBlockRate EncodeFblkRate(int fbr_raw) {
  switch (fbr_raw) {
    case 1:
      return FusedBlockRate::FBR_X1;
    case 2:
      return FusedBlockRate::FBR_X2;
    case 4:
      return FusedBlockRate::FBR_X4;
    case 8:
      return FusedBlockRate::FBR_X8;
    case 16:
      return FusedBlockRate::FBR_X16;
    default:
      LOG(ERROR) << "Not Supported FusedBlockRate : " << fbr_raw;
      exit(1);
  }
}

static inline constexpr U32 GetFletcher32(const U16* ptr, int len) {
  U32 sum0 = 0;
  U32 sum1 = 0;

  for (int i = 0; i < len; ++i) {
    sum1 += sum0 += ptr[i];
    sum0 = (sum0 & 0xFFFF) + (sum0 >> 16);
    sum1 = (sum1 & 0xFFFF) + (sum1 >> 16);
  }
  sum0 = (sum0 & 0xFFFF) + (sum0 >> 16);
  sum1 = (sum1 & 0xFFFF) + (sum1 >> 16);

  return (sum1 << 16) | sum0;
}

template <typename... Args>
constexpr std::string GetFmtString(const char* fmt, const Args&... args) {
  char buf[1024];
  snprintf(buf, sizeof(buf), fmt, args...);
  return std::string(buf);
}

/*----------------------------------------------------------------------------
 * Debugging helpers
 *----------------------------------------------------------------------------*/
__attribute__((unused)) static void __AIXV_CHECK_FAILED_PRE(const char* file,
                                                            int line,
                                                            const char* cond) {
  const char* f = __builtin_strrchr(file, '/');
  fprintf(stderr, "%s:%d] Check failed: %s ", f ? f + 1 : file, line, cond);
}

__attribute__((unused)) static void __AIXV_CHECK_FAILED_MSG() {
  fprintf(stderr, "\n");
  fflush(stderr);
  abort();
}

template <typename T0, typename T1>
static void __AIXV_CHECK_FAILED_VAL(const T0& v0, const T1& v1) {
  std::stringstream os0, os1;
  os0 << v0;
  os1 << v1;
  fprintf(stderr, "(%s vs. %s) ", os0.str().c_str(), os1.str().c_str());
}

template <typename... Types>
static void __AIXV_CHECK_FAILED_MSG(const Types&... args) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
  fprintf(stderr, args...);
#pragma GCC diagnostic pop
  fprintf(stderr, "\n");
  fflush(stderr);
  abort();
}

#define AIXV_CHECK(cond, ...)                             \
  do {                                                    \
    if (AIXV_PREDICT_FALSE(!(cond))) {                    \
      __AIXV_CHECK_FAILED_PRE(__FILE__, __LINE__, #cond); \
      __AIXV_CHECK_FAILED_MSG(__VA_ARGS__);               \
    }                                                     \
  } while (0)

#define __AIXV_CHECK_OP(op, v0, v1, ...)                                \
  do {                                                                  \
    if (AIXV_PREDICT_FALSE(!((v0)op(v1)))) {                            \
      __AIXV_CHECK_FAILED_PRE(__FILE__, __LINE__, #v0 " " #op " " #v1); \
      __AIXV_CHECK_FAILED_VAL(v0, v1);                                  \
      __AIXV_CHECK_FAILED_MSG(__VA_ARGS__);                             \
    }                                                                   \
  } while (0)

#define AIXV_CHECK_EQ(v0, v1, ...) __AIXV_CHECK_OP(==, v0, v1, __VA_ARGS__)
#define AIXV_CHECK_NE(v0, v1, ...) __AIXV_CHECK_OP(!=, v0, v1, __VA_ARGS__)
#define AIXV_CHECK_LE(v0, v1, ...) __AIXV_CHECK_OP(<=, v0, v1, __VA_ARGS__)
#define AIXV_CHECK_LT(v0, v1, ...) __AIXV_CHECK_OP(<, v0, v1, __VA_ARGS__)
#define AIXV_CHECK_GE(v0, v1, ...) __AIXV_CHECK_OP(>=, v0, v1, __VA_ARGS__)
#define AIXV_CHECK_GT(v0, v1, ...) __AIXV_CHECK_OP(>, v0, v1, __VA_ARGS__)

}  // namespace x330
