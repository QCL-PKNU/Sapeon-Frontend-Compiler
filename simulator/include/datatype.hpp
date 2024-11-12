#ifndef DATATYPE_HPP
#define DATATYPE_HPP

#include <string>

namespace dty {

enum class DataType {
  AIX_DATA_FLOAT = 0,
  FP32 = AIX_DATA_FLOAT,

  AIX_DATA_DOUBLE = 1,
  FP64 = AIX_DATA_DOUBLE,

  AIX_DATA_HALF = 2,
  FP16 = AIX_DATA_HALF,

  AIX_DATA_UINT8 = 3,
  UINT8 = AIX_DATA_UINT8,
  I8U = AIX_DATA_UINT8,

  AIX_DATA_SINT8 = 4,
  INT8 = AIX_DATA_SINT8,
  SINT8 = AIX_DATA_SINT8,
  I8 = AIX_DATA_SINT8,

  AIX_DATA_SINT16 = 5,
  INT16 = AIX_DATA_SINT16,
  SINT16 = AIX_DATA_SINT16,

  // FIXME: not implemented in proto
  INT32 = 6,
  INT64 = 7,

  I10U,

  NF8,
  NF8U,
  NF8E,

  NF9E,
  NF10,
  NF10U,
  NF12,
  NF13,
  NF13E,
  NF14,
  NF14E,
  NF15E,

  NF16,

  BF16
};

size_t SizeOf(DataType dtype);
std::string NameOf(DataType dtype);

template <typename Type>
DataType GetDataType();

std::ostream& operator<<(std::ostream& out, DataType dtype);

}  // namespace dty

#endif  // DATATYPE_HPP
