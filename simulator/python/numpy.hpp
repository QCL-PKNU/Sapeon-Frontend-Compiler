#pragma once

#include <memory>

#include "network/tensor.hpp"
#include "pybind11/pybind11.h"

namespace spgraph_simulator {
namespace python {

bool IsNumericNumpyArray(const pybind11::object& o);

std::unique_ptr<Tensor> FromNumpy(const pybind11::object& o);

}  // namespace python
}  // namespace spgraph_simulator
