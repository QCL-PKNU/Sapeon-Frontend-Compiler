#ifndef DUMP_DUMP_PATH_HELPER_HPP
#define DUMP_DUMP_PATH_HELPER_HPP

#include <optional>
#include <string>

#include "arguments.hpp"
#include "enums/dump.hpp"

class DumpPathHelper {
 public:
  DumpPathHelper(Arguments &args);
  std::string GetLayerOutputPath(int idx_layer);
  std::string GetActivationIntOutputPath(int idx_layer);
  std::string GetActivationFloatOutputPath(int idx_layer);
  std::string GetFilePath(const std::string &file_path);

 private:
  DumpLevel dump_level_;
  std::optional<std::string> dump_dir_;
};

#endif  // DUMP_DUMP_PATH_HELPER_HPP
