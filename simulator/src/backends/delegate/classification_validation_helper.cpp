#include "backends/delegate/classification_validation_helper.hpp"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
using std::string;
#include <vector>
using std::vector;

#include "datatype.hpp"
#include "enums/error.hpp"
#include "glog/logging.h"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "operations/cpu_operation.hpp"
#include "tl/expected.hpp"
using tl::expected;
using tl::make_unexpected;
#include "utility.hpp"

namespace fs = std::filesystem;

namespace validation {

ClassificationValidationHelper::ClassificationValidationHelper() {}

expected<bool, SimulatorError>
ClassificationValidationHelper::ValidateTopOneIndex(
    const std::vector<std::string>& class_dirs,
    const std::string& valid_image_path, std::shared_ptr<Tensor> output) {
  auto result = GetInferencedTopOneIndex(output);
  if (!result.has_value()) {
    return make_unexpected(result.error());
  }

  const size_t answer_idx = GetAnswerIndex(class_dirs, valid_image_path);

  size_t inferenced_idx = result.value();

  // if output is [1001], first index is no answer
  if (output->dimension().size() == 1001) {
    if (inferenced_idx == 0) {
      return false;
    }
    inferenced_idx--;
  }

  if (inferenced_idx == answer_idx) {
    return true;
  } else {
    return false;
  }
}

std::vector<std::string> ClassificationValidationHelper::GetClassDirs(
    const std::string& valid_image_dir) {
  fs::path dir = fs::current_path();
  dir /= valid_image_dir;

  std::vector<std::string> dirs;

  for (auto& entry : fs::directory_iterator(dir)) {
    if (entry.is_directory()) {
      dirs.push_back(entry.path().string());
    }
  }
  std::sort(dirs.begin(), dirs.end());
  return dirs;  // NRVO
}

size_t ClassificationValidationHelper::GetAnswerIndex(
    const std::vector<std::string>& valid_class_dirs,
    const std::string& valid_image_path) {
  fs::path file_path{valid_image_path};
  auto dir_path = file_path.parent_path().string();

  auto original =
      std::find(valid_class_dirs.begin(), valid_class_dirs.end(), dir_path);

  size_t index = original - valid_class_dirs.begin();
  LOG(INFO) << valid_image_path << ", "
            << fs::path{*original}.filename().string();
  return index;
}

expected<size_t, SimulatorError>
ClassificationValidationHelper::GetInferencedTopOneIndex(
    std::shared_ptr<Tensor> output) {
  if (output->dtype() == dty::DataType::FP32) {
    return GetInferencedTopOneIndex<float>(output);
  } else if (output->dtype() == dty::DataType::SINT8) {
    return GetInferencedTopOneIndex<int8_t>(output);
  }
  LOG(ERROR) << "GetInferencedTopOneIndex : validation is not implemented for: "
             << dty::NameOf(output->dtype());
  return make_unexpected(SimulatorError::kValidationError);
}

template <typename Type>
size_t ClassificationValidationHelper::GetInferencedTopOneIndex(
    std::shared_ptr<Tensor> output) {
  Type* data = output->data<Type>();
  const size_t size = output->dimension().size();

  Type* max = std::max_element(data, data + size);
  size_t index = std::find(data, data + size, *max) - data;
  return index;
}
}  // namespace validation
