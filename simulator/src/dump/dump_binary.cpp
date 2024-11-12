#include "dump/dump_binary.hpp"

#define CLASS DumpBinary
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
SCOPE::DumpBinary(DumpLevel dump_level, const string &dump_path)
    : dump_level_(dump_level), dump_path_(dump_path) {}

template <typename Type>
void SCOPE::DumpTensor(shared_ptr<Tensor> tensor, DumpLevel dump_level,
                       optional<int> precision) {
  if (dump_level > dump_level_ || tensor == nullptr) return;
  assert(dty::GetDataType<Type>() == tensor->dtype());
  Type *data = tensor->data<Type>();

  ofstream dump_file(dump_path_, std::ios::binary);
  dump_file.write(reinterpret_cast<const char *>(&data[0]), tensor->size());
}

template <typename Type>
void SCOPE::DumpTensorNHWC(shared_ptr<Tensor> tensor, DumpLevel dump_level,
                           optional<int> precision) {
  if (dump_level > dump_level_ || tensor == nullptr) return;
  assert(dty::GetDataType<Type>() == tensor->dtype());

  ofstream dump_file(dump_path_, std::ios::binary);

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
          Type value =
              data[n * offset_n + c * offset_c + h * offset_h + w * offset_w];
          dump_file.write(reinterpret_cast<const char *>(&value), sizeof(Type));
        }
}

template <typename Type>
void SCOPE::DumpVector(vector<Type> &vec, DumpLevel dump_level,
                       optional<int> precision) {
  if (dump_level > dump_level_ || vec.size() == 0) return;

  ofstream dump_file(dump_path_);
  dump_file.write(reinterpret_cast<const char *>(&vec[0]),
                  vec.size() * sizeof(Type));
}

template <typename Type>
void SCOPE::DumpAppendSingle(Type value, DumpLevel dump_level,
                             optional<int> precision) {
  if (dump_level > dump_level_) return;
  ofstream dump_file(dump_path_, std::ios::binary | std::ios::app);
  dump_file.write(reinterpret_cast<const char *>(&value), sizeof(Type));
  dump_file.close();
}

template class DumpBinary<double>;
template class DumpBinary<float>;
template class DumpBinary<int>;
template class DumpBinary<int8_t>;
template class DumpBinary<uint8_t>;
template class DumpBinary<int16_t>;
template class DumpBinary<int64_t>;
