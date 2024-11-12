#ifndef ENUMS_DUMP_HPP
#define ENUMS_DUMP_HPP

#include <string>

enum class DumpLevel { DUMP_NONE, DUMP_DEFAULT, DUMP_DEBUG };
enum class DumpFormat { DUMP_HEX, DUMP_OUTPUT, DUMP_SPACE, DUMP_BINARY };

DumpLevel GetDumpLevel(const std::string &level);

#endif  // ENUMS_DUMP_HPP
