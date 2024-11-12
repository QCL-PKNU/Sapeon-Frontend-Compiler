#ifndef CALIBRATION_MAX_HPP
#define CALIBRATION_MAX_HPP

#include <memory>

#include "enums/error.hpp"
#include "network/tensor.hpp"
#include "tl/expected.hpp"

namespace spgraph_simulator {
namespace calibration {
class Max {
 public:
  Max();
  tl::expected<void, SimulatorError> Set(const Tensor& tensor);
  float value();

 private:
  float value_;
};
}  // namespace calibration
}  // namespace spgraph_simulator

#endif  // CALIBRATION_MAX_HPP
