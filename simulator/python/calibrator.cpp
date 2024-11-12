#include "calibrator.hpp"

#include <filesystem>
#include <functional>

#include "calibration/calibrator.hpp"
#include "enums/calibration.hpp"
#include "numpy.hpp"
#include "simulator_api.hpp"

namespace spgraph_simulator {
namespace python {

namespace py = pybind11;
namespace fs = std::filesystem;

namespace {

//! FIXME: This utility should be replaced with the verified library
//! implementation.
class ScopedGuard {
 public:
  ScopedGuard() = delete;
  ScopedGuard(std::function<void()>&& cleanup) : cleanup_(std::move(cleanup)) {}
  ~ScopedGuard() { cleanup_(); }

 private:
  std::function<void()> cleanup_;
};

void ConvertOnnxToSapeon(const std::string& onnx_file_path) {
  const std::string compile_command =
      "onnx2sapeon --input " + onnx_file_path + " --calib dummy";

  if (std::system(compile_command.c_str()) != 0) {
    throw std::runtime_error("SAPEON graph Compilation failed");
  }
}
}  // namespace

void PyCalibrator::Collect(const py::object& ndarray) {
  if (!IsNumericNumpyArray(ndarray)) {
    throw std::runtime_error("given argument is not a numpy array");
  }
  auto tensor_ptr1 = FromNumpy(ndarray);
  p_calib_->Collect(*tensor_ptr1);
}

std::unique_ptr<calibration::Calibrator::CalibrationResult>
PyCalibrator::ComputeRange() {
  return p_calib_->ComputeRange();
}

void init_calibrator(py::module_ m) {
  py::enum_<calibration::CalibrationMethod>(m, "CalibrationMethod")
      .value("Max", calibration::CalibrationMethod::kMax)
      .value("Percentile", calibration::CalibrationMethod::kPercentile)
      .value("Entropy", calibration::CalibrationMethod::kEntropy)
      .value("Entropy2", calibration::CalibrationMethod::kEntropy2);

  py::class_<calibration::Calibrator::CalibrationResult>(m, "CalibrationResult")
      .def_readwrite("ranges",
                     &calibration::Calibrator::CalibrationResult::ranges)
      .def("as_textfile",
           &calibration::Calibrator::CalibrationResult::AsTextfile);

  py::class_<PyCalibrator>(m, "Calibrator")
      .def("collect", &PyCalibrator::Collect)
      .def("compute_range", &PyCalibrator::ComputeRange);

  m.def("make_calibrator",
        [](std::string model_path, calibration::CalibrationMethod calib_method,
           float percentile) {
          ConvertOnnxToSapeon(model_path);

          static const std::string sp_path = "spear_1-1.sp";
          ScopedGuard guard([=]() { fs::remove(sp_path); });

          auto calib = MakeCalibrator(sp_path, calib_method, percentile);

          return PyCalibrator(std::move(calib));
        });
}

}  // namespace python
}  // namespace spgraph_simulator
