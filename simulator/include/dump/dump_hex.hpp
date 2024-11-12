#ifndef DUMP_DUMP_HEX_HPP
#define DUMP_DUMP_HEX_HPP

#include <memory>
#include <optional>
#include <string>

#include "dump/dump.hpp"
#include "enums/dump.hpp"
#include "network/tensor.hpp"

template <typename Type>
class DumpHex : public Dump<Type> {
 public:
  DumpHex(DumpLevel dump_level, const std::string &dump_path);
  void DumpTensor(std::shared_ptr<Tensor> tensor, DumpLevel dump_level,
                  std::optional<int> precision = std::nullopt) override;
  void DumpTensorNHWC(std::shared_ptr<Tensor> tensor, DumpLevel dump_level,
                      std::optional<int> precision = std::nullopt) override;
  void DumpVector(std::vector<Type> &vec, DumpLevel dump_level,
                  std::optional<int> precision = std::nullopt) override;
  void DumpAppendSingle(Type value, DumpLevel dump_level,
                        std::optional<int> precision = std::nullopt) override;

 private:
  DumpLevel dump_level_;
  std::string dump_path_;
};

#endif  // DUMP_DUMP_HEX_HPP
