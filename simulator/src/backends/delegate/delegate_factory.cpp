#include "backends/delegate/delegate_factory.hpp"

#include <memory>

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "backends/delegate/calibration_delegate.hpp"
#include "backends/delegate/collect_delegate.hpp"
#include "backends/delegate/default_calibration_delegate.hpp"
#include "backends/delegate/dummy_calibration_delegate.hpp"
#include "backends/delegate/dummy_collect_delegate.hpp"
#include "backends/delegate/dummy_inference_delegate.hpp"
#include "backends/delegate/dummy_quantization_delegate.hpp"
#include "backends/delegate/dummy_validation_delegate.hpp"
#include "backends/delegate/fp_inference_delegate.hpp"
#include "backends/delegate/fp_validation_delegate.hpp"
#include "backends/delegate/inference_delegate.hpp"
#include "backends/delegate/quantization_delegate.hpp"
#include "backends/delegate/validation_delegate.hpp"
#include "backends/delegate/x220_inference_delegate.hpp"
#include "backends/delegate/x220_quantization_delegate.hpp"
#include "backends/delegate/x220_validation_delegate.hpp"
#include "backends/delegate/x330_collect_delegate.hpp"
#include "backends/delegate/x330_inference_delegate.hpp"
#include "backends/delegate/x330_quantization_delegate.hpp"
#include "backends/delegate/x330_validation_delegate.hpp"
#include "glog/logging.h"

std::unique_ptr<InferenceDelegate> DelegateFactory::GetInferenceDelegate(
    Backend &parent, Arguments &args) {
  if (!args.do_infer()) {
    return std::make_unique<DummyInferenceDelegate>(parent, args);
  }
  if (args.do_quant()) {
    if (args.quant_simulator().value() == "x220") {
      return std::make_unique<X220InferenceDelegate>(parent, args);
    } else if (args.quant_simulator().value() == "x330") {
      return std::make_unique<X330InferenceDelegate>(parent, args);
    } else {
      LOG(ERROR) << "Not Supported Quant Simulator";
      exit(1);
    }
  } else {
    return std::make_unique<FPInferenceDelegate>(parent, args);
  }
}

std::unique_ptr<CalibrationDelegate> DelegateFactory::GetCalibrationDelegate(
    Backend &parent, Arguments &args) {
  if (args.do_calib()) {
    return std::make_unique<DefaultCalibrationDelegate>(parent, args);
  } else {
    return std::make_unique<DummyCalibrationDelegate>(parent, args);
  }
}

std::unique_ptr<collect::CollectDelegate> DelegateFactory::GetCollectDelegate(
    Backend &parent, Arguments &args) {
  if (args.do_collect()) {
    return std::make_unique<collect::X330CollectDelegate>(parent, args);
  } else {
    return std::make_unique<collect::DummyCollectDelegate>(parent, args);
  }
}

std::unique_ptr<quantization::QuantizationDelegate>
DelegateFactory::GetQuantizationDelegate(Backend &parent, Arguments &args) {
  if (args.do_quant()) {
    if (args.quant_simulator().value() == "x220") {
      return std::make_unique<quantization::X220QuantizationDelegate>(parent,
                                                                      args);
    } else if (args.quant_simulator().value() == "x330") {
      return std::make_unique<quantization::X330QuantizationDelegate>(parent,
                                                                      args);
    } else {
      LOG(ERROR) << "Not Supported Quant Simulator";
      exit(1);
    }
  } else {
    return std::make_unique<quantization::DummyQuantizationDelegate>(parent,
                                                                     args);
  }
}

std::unique_ptr<validation::ValidationDelegate>
DelegateFactory::GetValidationDelegate(Backend &parent, Arguments &args) {
  using namespace validation;
  if (!args.do_valid()) {
    return std::make_unique<DummyValidationDelegate>(parent, args);
  }
  if (args.do_quant()) {
    if (args.quant_simulator().value() == "x220") {
      return std::make_unique<X220ValidationDelegate>(parent, args);
    } else if (args.quant_simulator().value() == "x330") {
      return std::make_unique<X330ValidationDelegate>(parent, args);
    } else {
      LOG(ERROR) << "Not Supported Quant Validation";
      exit(1);
    }
  } else {
    return std::make_unique<FPValidationDelegate>(parent, args);
  }
}
