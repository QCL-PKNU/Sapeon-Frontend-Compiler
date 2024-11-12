#include "cudnn/ops/activations.hpp"

#define BASE CudnnOperation
#define NAME Activations
#define CLASS Cudnn::NAME
#define SCOPE CLASS<Type, DataType>
#define DB double
#define FL float
#define UC uint8_t
#define SC int8_t
#define FP64 DB, CUDNN_DATA_DOUBLE
#define FP32 FL, CUDNN_DATA_FLOAT
#define FP16 FL, CUDNN_DATA_HALF
#define UINT8 UC, CUDNN_DATA_UINT8
#define INT8 SC, CUDNN_DATA_INT8
#define STR(x) #x
#define GET_STR(x) STR(x)

#include <memory>
using std::make_unique;
using std::unique_ptr;
#include <string>
using std::string;
#include <cudnn.h>

#include "factory.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cudnn_operation.hpp"

static bool kRegistered = Factory<BASE<DB>>::RegisterCreateFunction(
                              GET_STR(NAME), CLASS<FP64>::Create) &&
                          Factory<BASE<FL>>::RegisterCreateFunction(
                              GET_STR(NAME), CLASS<FP32>::Create) &&
                          Factory<BASE<SC>>::RegisterCreateFunction(
                              GET_STR(NAME), CLASS<INT8>::Create) &&
                          Factory<BASE<UC>>::RegisterCreateFunction(
                              GET_STR(NAME), CLASS<UINT8>::Create);

template <typename Type, cudnnDataType_t DataType>
unique_ptr<BASE<Type>> SCOPE::Create() {
  return make_unique<CLASS<Type, DataType>>();
}

template <typename Type, cudnnDataType_t DataType>
std::shared_ptr<Tensor> SCOPE::Forward(cudnnHandle_t &handle, Layer &layer) {
  string name = layer.activation_type();
  auto p_activation = Factory<CudnnOperation<Type>>::CreateInstance(name);

  output_ = p_activation.get()->Forward(handle, layer);

  return output_;
}
