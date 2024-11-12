#ifndef DUMP_DUMP_HPP
#define DUMP_DUMP_HPP

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "enums/dump.hpp"
#include "network/tensor.hpp"

template <typename Type>
class Dump {
 public:
  virtual void DumpTensor(std::shared_ptr<Tensor> tensor, DumpLevel dump_level,
                          std::optional<int> precision = std::nullopt) = 0;
  virtual void DumpTensorNHWC(std::shared_ptr<Tensor> tensor,
                              DumpLevel dump_level,
                              std::optional<int> precision = std::nullopt) = 0;
  virtual void DumpVector(std::vector<Type> &vec, DumpLevel dump_level,
                          std::optional<int> precision = std::nullopt) = 0;
  virtual void DumpAppendSingle(
      Type value, DumpLevel dump_level,
      std::optional<int> precision = std::nullopt) = 0;
  virtual ~Dump() {}
};

#endif  // DUMP_DUMP_HPP
