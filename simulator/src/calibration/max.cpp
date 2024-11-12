#include "calibration/max.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <string>

#include "datatype.hpp"
#include "enums/error.hpp"
#include "glog/logging.h"
#include "network/tensor.hpp"
#include "tl/expected.hpp"

namespace spgraph_simulator {
namespace calibration {
Max::Max() : value_(std::numeric_limits<float>::lowest()) {}

tl::expected<void, SimulatorError> Max::Set(const Tensor& tensor) {
  auto dtype = tensor.dtype();
  auto size = tensor.dimension().size();

  float min, max;

  switch (dtype) {
    case dty::DataType::FP32: {
      const float* data = tensor.data<float>();
      min = *std::min_element(data, data + size);
      max = *std::max_element(data, data + size);
      break;
    }
    case dty::DataType::FP64: {
      const double* data = tensor.data<double>();
      min = *std::min_element(data, data + size);
      max = *std::max_element(data, data + size);
      break;
    }
    default: {
      const std::string msg =
          "`" + dty::NameOf(dtype) + "` is not supported for calibration";
      LOG(ERROR) << msg;
      return tl::make_unexpected(SimulatorError::kCalibrationError);
    }
  }

  auto new_max = std::max(std::abs(min), std::abs(max));
  value_ = std::max(new_max, value_);

  return {};
}

float Max::value() { return value_; }
}  // namespace calibration
}  // namespace spgraph_simulator
