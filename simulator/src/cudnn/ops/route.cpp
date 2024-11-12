#include "cudnn/ops/route.hpp"

#define BASE CudnnOperation
#define NAME Route
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

#include <cassert>
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
#include "glog/logging.h"
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
std::shared_ptr<Tensor> SCOPE::Forward(cudnnHandle_t &handle, Layer &layer) {
  num_inputs_ = layer.predecessors().size();
  assert(num_inputs_ >= 1);
  inputs_ = layer.inputs();
  bool check = true;
  for (int i = 1; i < num_inputs_; ++i) {
    if (inputs_.at(i)->dtype() != inputs_.at(i - 1)->dtype()) {
      check = false;
    }
  }
  assert(check);

  InitOutputTensor();
  OperationForward();

  return output_;
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::InitOutputTensor() {
  for (int i = 0; i < num_inputs_; i++) {
    num_channels_ += inputs_[i]->c();
  }
  output_ =
      std::make_shared<Tensor>(inputs_[0]->n(), num_channels_, inputs_[0]->h(),
                               inputs_[0]->w(), dty::GetDataType<Type>());
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::OperationForward() {
  Type *input_data;
  Type *output_data = output_->data<Type>();

  size_t offset = 0;
  for (int i = 0; i < num_inputs_; i++) {
    input_data = inputs_[i]->data<Type>();
    const int offset_h = inputs_[i]->w();
    const int offset_c = inputs_[i]->h() * offset_h;
    const int offset_n = inputs_[i]->c() * offset_c;
    const int out_offset_n = num_channels_ * offset_c;
    for (int n = 0; n < inputs_[i]->n(); ++n)
      for (int c = 0; c < inputs_[i]->c(); ++c)
        for (int h = 0; h < inputs_[i]->h(); ++h)
          for (int w = 0; w < inputs_[i]->w(); ++w) {
            size_t idx_in = n * offset_n + c * offset_c + h * offset_h + w;
            size_t idx_out =
                n * out_offset_n + (c + offset) * offset_c + h * offset_h + w;
            output_data[idx_out] = input_data[idx_in];
          }
    offset += inputs_[i]->c();
  }
}
