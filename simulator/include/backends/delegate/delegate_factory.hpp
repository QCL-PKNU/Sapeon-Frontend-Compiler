#pragma once

#include <memory>

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "backends/delegate/calibration_delegate.hpp"
#include "backends/delegate/collect_delegate.hpp"
#include "backends/delegate/inference_delegate.hpp"
#include "backends/delegate/quantization_delegate.hpp"
#include "backends/delegate/validation_delegate.hpp"

class DelegateFactory {
 public:
  std::unique_ptr<InferenceDelegate> GetInferenceDelegate(Backend &parent,
                                                          Arguments &args);
  std::unique_ptr<CalibrationDelegate> GetCalibrationDelegate(Backend &parent,
                                                              Arguments &args);
  std::unique_ptr<collect::CollectDelegate> GetCollectDelegate(Backend &parent,
                                                               Arguments &args);
  std::unique_ptr<quantization::QuantizationDelegate> GetQuantizationDelegate(
      Backend &parent, Arguments &args);
  std::unique_ptr<validation::ValidationDelegate> GetValidationDelegate(
      Backend &parent, Arguments &args);
};
