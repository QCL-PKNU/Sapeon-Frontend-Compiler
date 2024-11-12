#include "cudnn/acts/leaky_relu.hpp"

#define BASE CudnnOperation
#define NAME LeakyReLU
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
using std::make_shared;
using std::make_unique;
using std::shared_ptr;
using std::unique_ptr;
#include <string>
using std::string;
#include <cudnn.h>

#include "datatype.hpp"
#include "factory.hpp"
#include "network/descriptor.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cudnn_operation.hpp"
#include "utility.hpp"

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
shared_ptr<Tensor> SCOPE::Forward(cudnnHandle_t &handle, Layer &layer) {
  input_ = layer.intermediate_activation() == nullptr
               ? layer.inputs(0)
               : layer.intermediate_activation();

  InitOutputTensor();
  ActivationForward();

  return output_;
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::InitOutputTensor() {
  output_ = std::make_shared<Tensor>(input_->n(), input_->c(), input_->h(),
                                     input_->w(), dty::GetDataType<Type>());
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::ActivationForward() {
  Type *in_data = input_->data<Type>();
  Type *out_data = output_->data<Type>();

  int each_height_size = output_->w();
  int each_channel_size = output_->h() * each_height_size;
  int each_batch_size = output_->c() * each_channel_size;

  for (int n = 0; n < output_->n(); ++n)
    for (int c = 0; c < output_->c(); ++c)
      for (int h = 0; h < output_->h(); ++h)
        for (int w = 0; w < output_->w(); ++w) {
          int idx = n * each_batch_size + c * each_channel_size +
                    h * each_height_size + w;

          out_data[idx] =
              (in_data[idx] > 0) ? in_data[idx] : in_data[idx] * 0.1;
        }
}
