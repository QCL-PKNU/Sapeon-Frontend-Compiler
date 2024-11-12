#include "dump/dump_factory.hpp"

#include <memory>
using std::make_unique;
using std::unique_ptr;
#include <string>
using std::string;

#include "arguments.hpp"
#include "dump/dump_binary.hpp"
#include "dump/dump_hex.hpp"
#include "dump/dump_output.hpp"
#include "dump/dump_space.hpp"
#include "enums/dump.hpp"
#include "glog/logging.h"

template <typename Type>
unique_ptr<Dump<Type>> DumpFactory::GetDump(DumpLevel level, DumpFormat fmt,
                                            const string& dump_path) {
  switch (fmt) {
    case DumpFormat::DUMP_OUTPUT:
      return make_unique<DumpOutput<Type>>(level, dump_path);
    case DumpFormat::DUMP_SPACE:
      return make_unique<DumpSpace<Type>>(level, dump_path);
    case DumpFormat::DUMP_HEX:
      return make_unique<DumpHex<Type>>(level, dump_path);
    case DumpFormat::DUMP_BINARY:
      return make_unique<DumpBinary<Type>>(level, dump_path);
    default:
      LOG(ERROR) << "Unknown dump-format!\n";
      exit(1);
  }
}

template unique_ptr<Dump<float>> DumpFactory::GetDump<float>(DumpLevel,
                                                             DumpFormat,
                                                             const string&);
template unique_ptr<Dump<double>> DumpFactory::GetDump<double>(DumpLevel,
                                                               DumpFormat,
                                                               const string&);
template unique_ptr<Dump<int8_t>> DumpFactory::GetDump<int8_t>(DumpLevel,
                                                               DumpFormat,
                                                               const string&);
template unique_ptr<Dump<uint8_t>> DumpFactory::GetDump<uint8_t>(DumpLevel,
                                                                 DumpFormat,
                                                                 const string&);
template unique_ptr<Dump<int16_t>> DumpFactory::GetDump<int16_t>(DumpLevel,
                                                                 DumpFormat,
                                                                 const string&);
template unique_ptr<Dump<int32_t>> DumpFactory::GetDump<int32_t>(DumpLevel,
                                                                 DumpFormat,
                                                                 const string&);
template unique_ptr<Dump<int64_t>> DumpFactory::GetDump<int64_t>(DumpLevel,
                                                                 DumpFormat,
                                                                 const string&);
