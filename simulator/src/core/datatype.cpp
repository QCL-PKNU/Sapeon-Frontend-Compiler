#include "datatype.hpp"

#include <iostream>
#include <string>
using std::string;

#include "enums/to_underlying_type.hpp"
#include "glog/logging.h"

namespace dty {

std::ostream& operator<<(std::ostream& out, DataType dtype) {
  return (out << spgraph_simulator::ToUnderlyingType(dtype));
}

size_t SizeOf(DataType dtype) {
  switch (dtype) {
    case DataType::FP32:
    case DataType::INT32:
      return 4;
    case DataType::FP64:
    case DataType::INT64:
      return 8;
    case DataType::FP16:
    case DataType::INT16:
      return 2;
    case DataType::UINT8:
    case DataType::SINT8:
      return 1;
    default:
      LOG(ERROR) << "Undefined Data Type! : " << dtype << "\n";
      exit(1);
  }
}

string NameOf(DataType dtype) {
  switch (dtype) {
    case DataType::FP32:
      return "FP32";
    case DataType::INT32:
      return "INT32";
    case DataType::FP64:
      return "FP64";
    case DataType::INT64:
      return "INT64";
    case DataType::FP16:
      return "FP16";
    case DataType::INT16:
      return "INT16";
    case DataType::UINT8:
      return "UINT8";
    case DataType::SINT8:
      return "SINT8";
    default:
      LOG(ERROR) << "Undefined Data Type! : " << dtype << "\n";
      exit(1);
  }
}

template <typename Type>
DataType GetDataType() {
  LOG(ERROR) << "Unknown Data Type! : " << typeid(Type).name() << '\n';
  exit(1);
}

template <>
DataType GetDataType<double>() {
  return DataType::FP64;
}

template <>
DataType GetDataType<float>() {
  return DataType::FP32;
}

template <>
DataType GetDataType<int16_t>() {
  return DataType::INT16;
}

template <>
DataType GetDataType<int8_t>() {
  return DataType::INT8;
}

template <>
DataType GetDataType<uint8_t>() {
  return DataType::UINT8;
}

template <>
DataType GetDataType<int32_t>() {
  return DataType::INT32;
}

template <>
DataType GetDataType<int64_t>() {
  return DataType::INT64;
}

template DataType GetDataType<double>();
template DataType GetDataType<float>();
template DataType GetDataType<int16_t>();
template DataType GetDataType<int8_t>();
template DataType GetDataType<uint8_t>();
template DataType GetDataType<int32_t>();
template DataType GetDataType<int64_t>();

}  // namespace dty
