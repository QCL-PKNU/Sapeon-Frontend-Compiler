#ifndef MXC_AIXH_COMMON_H
#define MXC_AIXH_COMMON_H

#include <stdint.h>
#include <stdlib.h>

#include <cstddef>
#include <vector>

#include "glog/logging.h"

namespace x220 {

/*----------------------------------------------------------------------------
 * Types
 *----------------------------------------------------------------------------*/
// Primitive types
typedef int8_t S8;
typedef int16_t S16;
typedef int32_t S32;
typedef int64_t S64;
typedef uint8_t U8;
typedef uint16_t U16;
typedef uint32_t U32;
typedef uint64_t U64;

// TensorCache block word type
typedef uint64_t TcbWord;

// TensorCache block half type
typedef uint32_t TcbHalf;

// // Data type codes
// #define DTY_UNSIGNED_BIT (4)
// #define DTY_AS_PRECISION(x) ((x) & ~DTY_UNSIGNED_BIT)
// #define DTY_AS_SIGNED(x) ((x) & ~DTY_UNSIGNED_BIT)
// #define DTY_AS_UNSIGNED(x) ((x) | DTY_UNSIGNED_BIT)
// #define DTY_IS_UNSIGNED(x) ((x)&DTY_UNSIGNED_BIT ? 1 : 0)

enum class DataType {
  DTY_INT4 = 0,
  DTY_INT8 = 1,
  DTY_INT16 = 2,
  DTY_MIX48 = 3,
  DTY_PREC_END = DTY_MIX48 + 1,

  DTY_SINT4 = DTY_INT4,
  DTY_SINT8 = DTY_INT8,
  DTY_SINT16 = DTY_INT16,
  DTY_SMIX48 = DTY_MIX48,
  DTY_UINT4 = DTY_INT4 | 4,
  DTY_UINT8 = DTY_INT8 | 4,
  DTY_UMIX48 = DTY_MIX48 | 4,
  DTY_TYPE_END = DTY_UMIX48 + 1
};

/*----------------------------------------------------------------------------
 * Utility functions
 *----------------------------------------------------------------------------*/
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

static inline S64 SignExtend(S64 x, int bit_width = 64) {
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

static inline int sizeof_dtype(DataType dty) {
  switch (dty) {
    case DataType::DTY_SINT8:
    case DataType::DTY_UINT8:
      return 1;
    case DataType::DTY_SINT16:
      return 2;
    default:
      return -1;
  }
}

static inline U32 GetFletcher32(const U16* ptr, int len) {
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

static inline void GetDataTypeMinMax(int& min, int& max, DataType dtype) {
  switch (dtype) {
    case DataType::DTY_UINT8:
      max = UINT8_MAX;
      min = 0;
      break;
    case DataType::DTY_SINT8:
      max = INT8_MAX;
      min = INT8_MIN;
      break;
    case DataType::DTY_SINT16:
      max = INT16_MAX;
      min = INT16_MIN;
      break;
    default:
      LOG(ERROR) << "Invalid dtype: " << sizeof_dtype(dtype) << '\n';
      exit(1);
  }
}

static inline bool IsUnsigned(DataType dtype) {
  return static_cast<int>(dtype) & 4;
}

}  // namespace x220

#endif  // MXC_AIXH_COMMON_H

// #include "common/aixh_config.h"
#ifndef MXC_SCALE_MANTISSA_BITS
#define MXC_SCALE_MANTISSA_BITS (15)
#endif  // MXC_SCALE_MANTISSA_BITS
