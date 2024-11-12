#include "enums/dump.hpp"

#include <string>
using std::string;

#include "glog/logging.h"

DumpLevel GetDumpLevel(const string &level) {
  if (level == "none") {
    return DumpLevel::DUMP_NONE;
  } else if (level == "default") {
    return DumpLevel::DUMP_DEFAULT;
  } else if (level == "debug") {
    return DumpLevel::DUMP_DEBUG;
  } else {
    LOG(ERROR) << "Undefined Dump Level! : " << level << "\n";
    exit(1);
  }
}
