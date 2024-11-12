#include "dump/dump_output.hpp"

#define CLASS DumpOutput
#define SCOPE CLASS<Type>

#include <cassert>
#include <fstream>
using std::ofstream;
#include <memory>
using std::shared_ptr;
#include <string>
using std::string;
using std::to_string;
#include <vector>
using std::vector;
#include <optional>
using std::optional;

#include "datatype.hpp"
#include "enums/dump.hpp"
#include "glog/logging.h"
#include "network/tensor.hpp"

template <typename Type>
SCOPE::DumpOutput(DumpLevel dump_level, const string &dump_path)
    : dump_level_(dump_level), dump_path_(dump_path) {
  // TODO: get precision and assign to member variable
  // TODO: check path, if file exist, truncate it
}

template <typename Type>
void SCOPE::DumpTensor(shared_ptr<Tensor> tensor, DumpLevel dump_level,
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

  for (size_t n = 0; n < tensor_n; ++n)
    for (size_t c = 0; c < tensor_c; ++c)
      for (size_t h = 0; h < tensor_h; ++h)
        for (size_t w = 0; w < tensor_w; ++w) {
          const size_t offset =
              n * offset_n + c * offset_c + h * offset_h + w * offset_w;
          dump_file << "output[" << offset << "] = " << to_string(data[offset])
                    << "\n";
        }
}

template <>
void DumpOutput<float>::DumpTensor(shared_ptr<Tensor> tensor,
                                   DumpLevel dump_level,
                                   optional<int> precision) {
  if (dump_level > dump_level_ || tensor == nullptr) return;
  assert(dty::GetDataType<float>() == tensor->dtype());
  assert(precision.has_value());

  ofstream dump_file(dump_path_);

  const size_t tensor_w = tensor->w();
  const size_t tensor_h = tensor->h();
  const size_t tensor_c = tensor->c();
  const size_t tensor_n = tensor->n();

  const size_t offset_w = 1;
  const size_t offset_h = tensor_w * offset_w;
  const size_t offset_c = tensor_h * offset_h;
  const size_t offset_n = tensor_c * offset_c;

  float *data = tensor->data<float>();

  dump_file.precision(precision.value());

  for (size_t n = 0; n < tensor_n; ++n)
    for (size_t c = 0; c < tensor_c; ++c)
      for (size_t h = 0; h < tensor_h; ++h)
        for (size_t w = 0; w < tensor_w; ++w) {
          const size_t offset =
              n * offset_n + c * offset_c + h * offset_h + w * offset_w;
          dump_file << "output[" << offset << "] = " << std::fixed
                    << data[offset] << "\n";
        }
}

template <>
void DumpOutput<double>::DumpTensor(shared_ptr<Tensor> tensor,
                                    DumpLevel dump_level,
                                    optional<int> precision) {
  if (dump_level > dump_level_ || tensor == nullptr) return;
  assert(dty::GetDataType<double>() == tensor->dtype());
  assert(precision.has_value());

  ofstream dump_file(dump_path_);

  const size_t tensor_w = tensor->w();
  const size_t tensor_h = tensor->h();
  const size_t tensor_c = tensor->c();
  const size_t tensor_n = tensor->n();

  const size_t offset_w = 1;
  const size_t offset_h = tensor_w * offset_w;
  const size_t offset_c = tensor_h * offset_h;
  const size_t offset_n = tensor_c * offset_c;

  double *data = tensor->data<double>();

  dump_file.precision(precision.value());

  for (size_t n = 0; n < tensor_n; ++n)
    for (size_t c = 0; c < tensor_c; ++c)
      for (size_t h = 0; h < tensor_h; ++h)
        for (size_t w = 0; w < tensor_w; ++w) {
          const size_t offset =
              n * offset_n + c * offset_c + h * offset_h + w * offset_w;
          dump_file << "output[" << offset << "] = " << std::fixed
                    << data[offset] << "\n";
        }
}

template <typename Type>
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

  for (size_t n = 0; n < tensor_n; ++n)
    for (size_t h = 0; h < tensor_h; ++h)
      for (size_t w = 0; w < tensor_w; ++w)
        for (size_t c = 0; c < tensor_c; ++c) {
          const size_t offset =
              n * offset_n + c * offset_c + h * offset_h + w * offset_w;
          dump_file << "output[" << offset << "] = " << to_string(data[offset])
                    << "\n";
        }
}

template <>
void DumpOutput<float>::DumpTensorNHWC(shared_ptr<Tensor> tensor,
                                       DumpLevel dump_level,
                                       optional<int> precision) {
  if (dump_level > dump_level_ || tensor == nullptr) return;
  assert(dty::GetDataType<float>() == tensor->dtype());
  assert(precision.has_value());

  ofstream dump_file(dump_path_);

  const size_t tensor_w = tensor->w();
  const size_t tensor_h = tensor->h();
  const size_t tensor_c = tensor->c();
  const size_t tensor_n = tensor->n();

  const size_t offset_w = 1;
  const size_t offset_h = tensor_w * offset_w;
  const size_t offset_c = tensor_h * offset_h;
  const size_t offset_n = tensor_c * offset_c;

  float *data = tensor->data<float>();

  dump_file.precision(precision.value());

  for (size_t n = 0; n < tensor_n; ++n)
    for (size_t h = 0; h < tensor_h; ++h)
      for (size_t w = 0; w < tensor_w; ++w)
        for (size_t c = 0; c < tensor_c; ++c) {
          const size_t offset =
              n * offset_n + c * offset_c + h * offset_h + w * offset_w;
          dump_file << "output[" << offset << "] = " << std::fixed
                    << data[offset] << "\n";
        }
}

template <>
void DumpOutput<double>::DumpTensorNHWC(shared_ptr<Tensor> tensor,
                                        DumpLevel dump_level,
                                        optional<int> precision) {
  if (dump_level > dump_level_ || tensor == nullptr) return;
  assert(dty::GetDataType<double>() == tensor->dtype());
  assert(precision.has_value());

  ofstream dump_file(dump_path_);

  const size_t tensor_w = tensor->w();
  const size_t tensor_h = tensor->h();
  const size_t tensor_c = tensor->c();
  const size_t tensor_n = tensor->n();

  const size_t offset_w = 1;
  const size_t offset_h = tensor_w * offset_w;
  const size_t offset_c = tensor_h * offset_h;
  const size_t offset_n = tensor_c * offset_c;

  double *data = tensor->data<double>();

  dump_file.precision(precision.value());

  for (size_t n = 0; n < tensor_n; ++n)
    for (size_t h = 0; h < tensor_h; ++h)
      for (size_t w = 0; w < tensor_w; ++w)
        for (size_t c = 0; c < tensor_c; ++c) {
          const size_t offset =
              n * offset_n + c * offset_c + h * offset_h + w * offset_w;
          dump_file << "output[" << offset << "] = " << std::fixed
                    << data[offset] << "\n";
        }
}

template <typename Type>
void SCOPE::DumpVector(vector<Type> &vec, DumpLevel dump_level,
                       optional<int> precision) {
  if (dump_level > dump_level_ || vec.size() == 0) return;

  ofstream dump_file(dump_path_);

  for (int i = 0; i < vec.size(); i++) {
    dump_file << "output[" << i << "] = " << to_string(vec[i]) << "\n";
  }
}

template <>
void DumpOutput<float>::DumpVector(vector<float> &vec, DumpLevel dump_level,
                                   optional<int> precision) {
  if (dump_level > dump_level_ || vec.size() == 0) return;
  assert(precision.has_value());

  ofstream dump_file(dump_path_);

  dump_file.precision(precision.value());

  for (int i = 0; i < vec.size(); i++) {
    dump_file << "output[" << i << "] = " << std::fixed << vec[i] << "\n";
  }
}

template <>
void DumpOutput<double>::DumpVector(vector<double> &vec, DumpLevel dump_level,
                                    optional<int> precision) {
  if (dump_level > dump_level_ || vec.size() == 0) return;
  assert(precision.has_value());
  ofstream dump_file(dump_path_);

  dump_file.precision(precision.value());

  for (int i = 0; i < vec.size(); i++) {
    dump_file << "output[" << i << "] = " << std::fixed << vec[i] << "\n";
  }
}

template <typename Type>
void SCOPE::DumpAppendSingle(Type value, DumpLevel dump_level,
                             optional<int> precision) {
  if (dump_level > dump_level_) return;
  ofstream dump_file(dump_path_, std::ios::app);

  // TODO: need to get index
  dump_file << "output[" << 0 << "] = " << to_string(value) << "\n";
}

template <>
void DumpOutput<float>::DumpAppendSingle(float value, DumpLevel dump_level,
                                         optional<int> precision) {
  if (dump_level > dump_level_) return;
  assert(precision.has_value());
  ofstream dump_file(dump_path_, std::ios::app);

  dump_file.precision(precision.value());

  // TODO: need to get index
  dump_file << "output[" << 0 << "] = " << std::fixed << value << "\n";
}

template <>
void DumpOutput<double>::DumpAppendSingle(double value, DumpLevel dump_level,
                                          optional<int> precision) {
  if (dump_level > dump_level_) return;
  assert(precision.has_value());
  ofstream dump_file(dump_path_, std::ios::app);

  dump_file.precision(precision.value());

  // TODO: need to get index
  dump_file << "output[" << 0 << "] = " << std::fixed << value << "\n";
}

template class DumpOutput<double>;
template class DumpOutput<float>;
template class DumpOutput<int>;
template class DumpOutput<int8_t>;
template class DumpOutput<uint8_t>;
template class DumpOutput<int16_t>;
template class DumpOutput<int64_t>;
