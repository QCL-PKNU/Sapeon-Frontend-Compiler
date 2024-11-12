#include "calibrator.hpp"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "simulator_api.hpp"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL spgraph_simulator_python_ARRAY_API
#include "numpy/arrayobject.h"

namespace spgraph_simulator {
namespace python {

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(simulator, m) {
  ([]() -> void { import_array1(); })();
  init_calibrator(m);
}

}  // namespace python
}  // namespace spgraph_simulator
