#include "dump/dump_hex.hpp"

#define CLASS DumpHex
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
SCOPE::DumpHex(DumpLevel dump_level, const string &dump_path)
    : dump_level_(dump_level), dump_path_(dump_path) {}

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

  dump_file << std::setfill('0') << std::hex << std::uppercase;

  for (size_t n = 0; n < tensor_n; ++n)
    for (size_t c = 0; c < tensor_c; ++c)
      for (size_t h = 0; h < tensor_h; ++h)
        for (size_t w = 0; w < tensor_w; ++w) {
          Type value =
              data[n * offset_n + c * offset_c + h * offset_h + w * offset_w];

          unsigned char *hex_data = reinterpret_cast<unsigned char *>(&value);
          for (int i = 0; i < sizeof(Type); i++) {
            dump_file << std::setw(2) << static_cast<int>(hex_data[i]);
          }
          dump_file << ",";  // csv
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

  dump_file << std::setfill('0') << std::hex << std::uppercase;

  for (size_t n = 0; n < tensor_n; ++n)
    for (size_t h = 0; h < tensor_h; ++h)
      for (size_t w = 0; w < tensor_w; ++w)
        for (size_t c = 0; c < tensor_c; ++c) {
          Type value =
              data[n * offset_n + c * offset_c + h * offset_h + w * offset_w];

          unsigned char *hex_data = reinterpret_cast<unsigned char *>(&value);
          for (int i = 0; i < sizeof(Type); i++) {
            dump_file << std::setw(2) << static_cast<int>(hex_data[i]);
          }
          dump_file << ",";  // csv
        }
}

template <typename Type>
void SCOPE::DumpVector(vector<Type> &vec, DumpLevel dump_level,
                       optional<int> precision) {
  if (dump_level > dump_level_ || vec.size() == 0) return;

  ofstream dump_file(dump_path_);

  dump_file << std::setfill('0') << std::hex << std::uppercase;

  for (auto &value : vec) {
    unsigned char *hex_data = reinterpret_cast<unsigned char *>(&value);
    for (int i = 0; i < sizeof(Type); i++) {
      dump_file << std::setw(2) << static_cast<int>(hex_data[i]);
    }
    dump_file << ",";  // csv
  }
}

template <typename Type>
void SCOPE::DumpAppendSingle(Type value, DumpLevel dump_level,
                             optional<int> precision) {
  if (dump_level > dump_level_) return;
  ofstream dump_file(dump_path_, std::ios::app);

  dump_file << std::setfill('0') << std::hex << std::uppercase;

  unsigned char *hex_data = reinterpret_cast<unsigned char *>(&value);
  for (int i = 0; i < sizeof(Type); i++) {
    dump_file << std::setw(2) << static_cast<int>(hex_data[i]);
  }
  dump_file << ",";  // csv
}

template class DumpHex<double>;
template class DumpHex<float>;
template class DumpHex<int>;
template class DumpHex<int8_t>;
template class DumpHex<uint8_t>;
template class DumpHex<int16_t>;
template class DumpHex<int64_t>;
