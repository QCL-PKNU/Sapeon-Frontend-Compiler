#ifndef DUMP_DUMP_SPACE_HPP
#define DUMP_DUMP_SPACE_HPP

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "dump/dump.hpp"
#include "enums/dump.hpp"
#include "network/tensor.hpp"

template <typename Type>
class DumpSpace : public Dump<Type> {
 public:
  DumpSpace(DumpLevel dump_level, const std::string &dump_path);
  void DumpTensor(std::shared_ptr<Tensor> tensor, DumpLevel dump_level,
                  std::optional<int> precision = std::nullopt) override;
  template <typename ChangeType>
  void DumpTensor(std::shared_ptr<Tensor> tensor, DumpLevel dump_level,
                  std::optional<int> precision = std::nullopt);
  void DumpTensorNHWC(std::shared_ptr<Tensor> tensor, DumpLevel dump_level,
                      std::optional<int> precision = std::nullopt) override;
  template <typename ChangeType>
  void DumpTensorNHWC(std::shared_ptr<Tensor> tensor, DumpLevel dump_level,
                      std::optional<int> precision = std::nullopt);
  void DumpVector(std::vector<Type> &vec, DumpLevel dump_level,
                  std::optional<int> precision = std::nullopt) override;
  template <typename ChangeType>
  void DumpVector(std::vector<Type> &vec, DumpLevel dump_level,
                  std::optional<int> precision = std::nullopt);
  void DumpAppendSingle(Type value, DumpLevel dump_level,
                        std::optional<int> precision = std::nullopt) override;
  template <typename ChangeType>
  void DumpAppendSingle(Type value, DumpLevel dump_level,
                        std::optional<int> precision = std::nullopt);

 private:
  DumpLevel dump_level_;
  std::string dump_path_;
};

#endif  // DUMP_DUMP_SPACE_HPP
