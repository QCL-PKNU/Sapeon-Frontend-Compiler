#include "backends/cudnn.hpp"

#define BASE Backend
#define NAME cudnn
#define CLASS CudnnBackend
#define OPERATION_CLASS CudnnOperation
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

#include <memory>
using std::make_unique;
using std::shared_ptr;
using std::unique_ptr;
#include <string>
using std::string;

#include <cudnn.h>

#include "glog/logging.h"
#include "tl/expected.hpp"
using tl::expected;
using tl::unexpected;

#include "backends/backend.hpp"
#include "enums/error.hpp"
#include "factory.hpp"
#include "operations/cudnn_operation.hpp"

static bool kRegistered =
    Factory<BASE>::RegisterCreateFunction(GET_STR(NAME), CLASS::CreateBackend);

unique_ptr<BASE> SCOPE::CreateBackend() { return make_unique<CLASS>(); }

SCOPE::CudnnBackend() {
  // Create cuDNN handle
  cudnnCreate(&handle_);
}

SCOPE::~CudnnBackend() {
  // Destroy cuDNN handle
  cudnnDestroy(handle_);
}

tl::expected<shared_ptr<Tensor>, SimulatorError> SCOPE::Forward(
    Layer &layer, const int idx_sublayer) {
  if (network_->input()->dtype() == dty::DataType::FP32) {
    return Forward<float>(layer, idx_sublayer);
  } else if (network_->input()->dtype() == dty::DataType::FP64) {
    return Forward<double>(layer, idx_sublayer);
  } else {
    const string msg =
        "Invalid Data Type: " + dty::NameOf(network_->input()->dtype());
    LOG(ERROR) << msg;
    return unexpected<SimulatorError>(SimulatorError::kInvalidDataType);
  }
}

template <typename Type>
shared_ptr<Tensor> SCOPE::Forward(Layer &layer, const int idx_sublayer) {
  string operation_name = layer.operation_types(idx_sublayer);
  auto p_operation =
      Factory<OPERATION_CLASS<Type>>::CreateInstance(operation_name);

  return p_operation->Forward(handle_, layer);
}
