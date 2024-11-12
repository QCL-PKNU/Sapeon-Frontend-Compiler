#include "dump/dump_space.hpp"

#define CLASS DumpSpace
#define SCOPE CLASS<Type>

#include <cassert>
#include <fstream>
using std::ofstream;
#include <iomanip>
#include <memory>
using std::shared_ptr;
#include <string>
using std::string;
#include <vector>
using std::vector;
#include <optional>
using std::optional;

#include "datatype.hpp"
#include "enums/dump.hpp"
#include "glog/logging.h"
#include "network/tensor.hpp"

template <typename Type>
SCOPE::DumpSpace(DumpLevel dump_level, const string &dump_path)
    : dump_level_(dump_level), dump_path_(dump_path) {}

template <typename Type>
template <typename ChangeType>
void SCOPE::DumpTensor(shared_ptr<Tensor> tensor, DumpLevel dump_level,
                       optional<int> precision) {
  if (dump_level > dump_level_ || tensor == nullptr) return;
  assert(dty::GetDataType<Type>() == tensor->dtype());

  ofstream dump_file(dump_path_);

  if (precision.has_value()) {
    dump_file << std::setprecision(precision.value());
  }

  const size_t tensor_w = tensor->w();
  const size_t tensor_h = tensor->h();
  const size_t tensor_c = tensor->c();
  const size_t tensor_n = tensor->n();

  const size_t offset_w = 1;
  const size_t offset_h = tensor_w * offset_w;
  const size_t offset_c = tensor_h * offset_h;
  const size_t offset_n = tensor_c * offset_c;

  Type *data = tensor->data<Type>();

  for (size_t n = 0; n < tensor_n; ++n)
    for (size_t c = 0; c < tensor_c; ++c)
      for (size_t h = 0; h < tensor_h; ++h)
        for (size_t w = 0; w < tensor_w; ++w) {
          dump_file << static_cast<ChangeType>(
                           data[n * offset_n + c * offset_c + h * offset_h +
                                w * offset_w])
                    << " ";  // space
        }
  dump_file << "\n";
}

template <typename Type>
void SCOPE::DumpTensor(shared_ptr<Tensor> tensor, DumpLevel dump_level,
                       optional<int> precision) {
  DumpTensor<Type>(tensor, dump_level, precision);
}

template <>
void DumpSpace<int8_t>::DumpTensor(shared_ptr<Tensor> tensor,
                                   DumpLevel dump_level,
                                   optional<int> precision) {
  DumpTensor<int64_t>(tensor, dump_level, precision);
}

template <>
void DumpSpace<uint8_t>::DumpTensor(shared_ptr<Tensor> tensor,
                                    DumpLevel dump_level,
                                    optional<int> precision) {
  DumpTensor<int64_t>(tensor, dump_level, precision);
}

template <typename Type>
template <typename ChangeType>
void SCOPE::DumpTensorNHWC(shared_ptr<Tensor> tensor, DumpLevel dump_level,
                           optional<int> precision) {
  if (dump_level > dump_level_ || tensor == nullptr) return;
  assert(dty::GetDataType<Type>() == tensor->dtype());

  ofstream dump_file(dump_path_);

  const size_t tensor_w = tensor->w();
  const size_t tensor_h = tensor->h();
  const size_t tensor_c = tensor->c();
  const size_t tensor_n = tensor->n();

  const size_t offset_w = 1;
  const size_t offset_h = tensor_w * offset_w;
  const size_t offset_c = tensor_h * offset_h;
  const size_t offset_n = tensor_c * offset_c;

  Type *data = tensor->data<Type>();

  if (precision.has_value()) {
    dump_file << std::setprecision(precision.value());
  }

  for (size_t n = 0; n < tensor_n; ++n)
    for (size_t h = 0; h < tensor_h; ++h)
      for (size_t w = 0; w < tensor_w; ++w)
        for (size_t c = 0; c < tensor_c; ++c) {
          dump_file << static_cast<ChangeType>(
                           data[n * offset_n + c * offset_c + h * offset_h +
                                w * offset_w])
                    << " ";  // space
        }
  dump_file << "\n";
}

template <typename Type>
void SCOPE::DumpTensorNHWC(shared_ptr<Tensor> tensor, DumpLevel dump_level,
                           optional<int> precision) {
  DumpTensorNHWC<Type>(tensor, dump_level, precision);
}

template <>
void DumpSpace<int8_t>::DumpTensorNHWC(shared_ptr<Tensor> tensor,
                                       DumpLevel dump_level,
                                       optional<int> precision) {
  DumpTensorNHWC<int64_t>(tensor, dump_level, precision);
}

template <>
void DumpSpace<uint8_t>::DumpTensorNHWC(shared_ptr<Tensor> tensor,
                                        DumpLevel dump_level,
                                        optional<int> precision) {
  DumpTensorNHWC<int64_t>(tensor, dump_level, precision);
}

template <typename Type>
template <typename ChangeType>
void SCOPE::DumpVector(vector<Type> &vec, DumpLevel dump_level,
                       optional<int> precision) {
  if (dump_level > dump_level_ || vec.size() == 0) return;

  ofstream dump_file(dump_path_);

  if (precision.has_value()) {
    dump_file << std::setprecision(precision.value());
  }

  for (const auto &value : vec) {
    dump_file << static_cast<ChangeType>(value) << " ";  // space
  }
}

template <typename Type>
void SCOPE::DumpVector(vector<Type> &vec, DumpLevel dump_level,
                       optional<int> precision) {
  DumpVector<Type>(vec, dump_level, precision);
}

template <>
void DumpSpace<uint8_t>::DumpVector(vector<uint8_t> &vec, DumpLevel dump_level,
                                    optional<int> precision) {
  DumpVector<int64_t>(vec, dump_level, precision);
}

template <>
void DumpSpace<int8_t>::DumpVector(vector<int8_t> &vec, DumpLevel dump_level,
                                   optional<int> precision) {
  DumpVector<int64_t>(vec, dump_level, precision);
}

template <typename Type>
template <typename ChangeType>
void SCOPE::DumpAppendSingle(Type value, DumpLevel dump_level,
                             optional<int> precision) {
  if (dump_level > dump_level_) return;
  ofstream dump_file(dump_path_, std::ios::app);

  if (precision.has_value()) {
    dump_file << std::setprecision(precision.value());
  }

  dump_file << static_cast<ChangeType>(value) << " ";  // space
}

template <typename Type>
void SCOPE::DumpAppendSingle(Type value, DumpLevel dump_level,
                             optional<int> precision) {
  DumpAppendSingle<Type>(value, dump_level, precision);
}

template <>
void DumpSpace<uint8_t>::DumpAppendSingle(uint8_t value, DumpLevel dump_level,
                                          optional<int> precision) {
  DumpAppendSingle<int64_t>(value, dump_level, precision);
}

template <>
void DumpSpace<int8_t>::DumpAppendSingle(int8_t value, DumpLevel dump_level,
                                         optional<int> precision) {
  DumpAppendSingle<int64_t>(value, dump_level, precision);
}

template class DumpSpace<double>;
template class DumpSpace<float>;
template class DumpSpace<int>;
template class DumpSpace<int8_t>;
template class DumpSpace<uint8_t>;
template class DumpSpace<int16_t>;
template class DumpSpace<int64_t>;
