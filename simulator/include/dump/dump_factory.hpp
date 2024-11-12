#ifndef DUMP_DUMP_FACTORY_HPP
#define DUMP_DUMP_FACTORY_HPP

#include <memory>
#include <string>

#include "arguments.hpp"
#include "dump/dump.hpp"
#include "enums/dump.hpp"

class DumpFactory {
 public:
  DumpFactory() {}
  template <typename Type>
  std::unique_ptr<Dump<Type>> GetDump(DumpLevel level, DumpFormat fmt,
                                      const std::string &dump_path);
};

#endif  // DUMP_DUMP_FACTORY_HPP
