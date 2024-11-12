#include "numpy.hpp"

#include <map>
#include <memory>

#include "glog/logging.h"
#include "network/tensor.hpp"
#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL spgraph_simulator_python_ARRAY_API
#include "numpy/arrayobject.h"

namespace spgraph_simulator {
namespace python {
namespace py = pybind11;

bool IsNumericNumpyType(int npy_type) {
  return npy_type < NPY_OBJECT || npy_type == NPY_HALF;
}

bool PyObjectCheck_NumpyArray(PyObject* o) {
  return (PyObject_HasAttrString(o, "__array_finalize__") != 0);
}

bool IsNumericNumpyArray(const py::object& obj) {
  if (PyObjectCheck_NumpyArray(obj.ptr())) {
    int npy_type = PyArray_TYPE(reinterpret_cast<PyArrayObject*>(obj.ptr()));
    return IsNumericNumpyType(npy_type);
  }

  return false;
}

dty::DataType NumpyTypeToSapeonType(int numpy_type) {
  //! FIXME: should cover all NPY types
  static std::map<int, dty::DataType> type_map{
      {NPY_INT8, dty::DataType::SINT8},
      {NPY_UINT8, dty::DataType::UINT8},
      {NPY_INT16,
       dty::DataType::SINT16}, /*{NPY_UINT16, dty::DataType::UINT16},*/
      {NPY_FLOAT16, dty::DataType::FP16},
      {NPY_FLOAT, dty::DataType::FP32}};

  const auto it = type_map.find(numpy_type);
  if (it == type_map.end()) {
    throw std::runtime_error(
        std::string("No corresponding Numpy type for Tensor Type.: ") +
        std::to_string(numpy_type));
  } else {
    return it->second;
  }
}

static std::vector<size_t> GetArrayShape(PyArrayObject* obj) {
  const int dim = PyArray_NDIM(obj);
  const npy_intp* npy_dims = PyArray_DIMS(obj);
  std::vector<size_t> dims;
  for (int i = 0; i < dim; ++i) {
    int x = *(npy_dims + i);
    dims.push_back(x);
  }
  if (dims.size() != 4) {
    throw std::runtime_error("dims must be 4");
  }

  return dims;
}

static void CopyDataToTensor(PyArrayObject* darray, int npy_type,
                             Tensor& tensor) {
  if (npy_type == NPY_UNICODE || npy_type == NPY_STRING ||
      npy_type == NPY_VOID || npy_type == NPY_OBJECT) {
    throw std::runtime_error("NYI");
  } else {
    void* buffer = tensor.data<uint8_t>();
    size_t len = tensor.size();
    memcpy(buffer, PyArray_DATA(darray), len);
  }
}

std::unique_ptr<Tensor> FromNumpy(const py::object& obj) {
  PyArrayObject* darray = reinterpret_cast<PyArrayObject*>(obj.ptr());
  if (!PyArray_ISCONTIGUOUS(darray) ||
      (darray = PyArray_GETCONTIGUOUS(darray)) == nullptr) {
    throw std::runtime_error("Require contiguous numpy array of values");
  }
  const auto dims = GetArrayShape(darray);

  const int npy_type = PyArray_TYPE(darray);
  const auto sapeon_dtype = NumpyTypeToSapeonType(npy_type);

  auto p_tensor = std::make_unique<Tensor>(dims, sapeon_dtype);
  CopyDataToTensor(darray, npy_type, *p_tensor);

  return p_tensor;
}

}  // namespace python
}  // namespace spgraph_simulator
