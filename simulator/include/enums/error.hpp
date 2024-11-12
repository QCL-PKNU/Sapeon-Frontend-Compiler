#ifndef ENUMS_ERROR_HPP
#define ENUMS_ERROR_HPP

enum class SimulatorError {
  kModelParsingError = 1,  // Parser failed to parse model binary
  kInvalidModel,           // Model is invalid
  kArgumentsParsingError,  // Failed to parse Arguments, or is invalid
  kOperationError,         // Failed to run operations(inference)
  kInvalidDataType,        // Data type of tensor is invalid
  kCreateInstanceError,    // Factory failed to create class instance
  kNumpyLoadError,         // Failed to load numpy image
  kTensorShapeError,       // Shape of tensor is invalid
  kFileReadError,          // Failed to read file
  kFileWriteError,         // Failed to write file
  kCalibrationError,       // Failed to run calibration
  kQuantizationError,      // Failed to run quantization
  kValidationError,        // Failed to run validation
};

#endif  // ENUMS_ERROR_HPP
