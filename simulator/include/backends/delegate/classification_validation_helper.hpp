#ifndef BACKENDS_DELEGATE_CLASSIFICATION_VALIDATION_HELPER_HPP
#define BACKENDS_DELEGATE_CLASSIFICATION_VALIDATION_HELPER_HPP

#include <memory>
#include <string>

#include "enums/error.hpp"
#include "network/network.hpp"
#include "tl/expected.hpp"

namespace validation {
class ClassificationValidationHelper {
 public:
  ClassificationValidationHelper();
  tl::expected<bool, SimulatorError> ValidateTopOneIndex(
      const std::vector<std::string> &class_dirs,
      const std::string &valid_image_path, std::shared_ptr<Tensor> output);
  std::vector<std::string> GetClassDirs(const std::string &valid_image_dir);

 private:
  size_t GetAnswerIndex(const std::vector<std::string> &valid_class_dirs,
                        const std::string &valid_image_path);
  tl::expected<size_t, SimulatorError> GetInferencedTopOneIndex(
      std::shared_ptr<Tensor> output);
  template <typename Type>
  size_t GetInferencedTopOneIndex(std::shared_ptr<Tensor> output);
};
}  // namespace validation
#endif  // BACKENDS_DELEGATE_CLASSIFICATION_VALIDATION_HELPER_HPP
