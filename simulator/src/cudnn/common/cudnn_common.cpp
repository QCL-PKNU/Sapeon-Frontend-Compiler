// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cudnn/common/cudnn_common.hpp"

#include "cudnn/common/inlined_containers_fwd.hpp"
#include "cudnn/common/utils.hpp"

// #include "core/common/inlined_containers.h"
// #include "core/providers/cpu/tensor/utils.h"
#include "gsl-lite.hpp"
// #include "shared_inc/cuda_call.h"

namespace Cudnn {

CudnnTensor::CudnnTensor() : tensor_(nullptr) {}

CudnnTensor::~CudnnTensor() {
  if (tensor_ != nullptr) {
    cudnnDestroyTensorDescriptor(tensor_);
    tensor_ = nullptr;
  }
}

bool CudnnTensor::CreateTensorIfNeeded() {
  if (!tensor_)
    assert(cudnnCreateTensorDescriptor(&tensor_) == CUDNN_STATUS_SUCCESS);
  return true;
}

bool CudnnTensor::Set(gsl::span<const int64_t> input_dims,
                      cudnnDataType_t dataType) {
  assert(CreateTensorIfNeeded());

  int rank = gsl::narrow_cast<int>(input_dims.size());
  TensorPitches pitches(input_dims);
  InlinedVector<int, kTensorShapeSmallBufferElementsSize> dims(rank);
  InlinedVector<int, kTensorShapeSmallBufferElementsSize> strides(rank);
  for (int i = 0; i < rank; i++) {
    dims[i] = gsl::narrow_cast<int>(input_dims[i]);
    strides[i] = gsl::narrow_cast<int>(pitches[i]);
  }
  assert(cudnnSetTensorNdDescriptor(tensor_, dataType, static_cast<int>(rank),
                                    dims.data(),
                                    strides.data()) == CUDNN_STATUS_SUCCESS);
  return true;
}

bool CudnnTensor::Set(const CudnnTensor& x_desc, cudnnBatchNormMode_t mode) {
  assert(CreateTensorIfNeeded());
  assert(cudnnDeriveBNTensorDescriptor(tensor_, x_desc, mode) ==
         CUDNN_STATUS_SUCCESS);
  return true;
}

CudnnDataTensor::CudnnDataTensor() : tensor_(nullptr) {}

CudnnDataTensor::~CudnnDataTensor() {
  if (tensor_ != nullptr) {
    cudnnDestroyRNNDataDescriptor(tensor_);
    tensor_ = nullptr;
  }
}

bool CudnnDataTensor::CreateTensorIfNeeded() {
  if (!tensor_)
    assert(cudnnCreateRNNDataDescriptor(&tensor_) == CUDNN_STATUS_SUCCESS);
  return true;
}

bool CudnnDataTensor::Set(cudnnDataType_t dataType, int64_t max_seq_length,
                          int64_t batch_size, int64_t data_size,
                          const int32_t* seq_lengths) {
  assert(CreateTensorIfNeeded());

  // CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED works with
  // CUDNN_RNN_PADDED_IO_ENABLED, so that it will auto fill 0 for the shorter
  // sequences
  cudnnRNNDataLayout_t layout = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED;
  float padding_fill = 0.0f;
  assert(cudnnSetRNNDataDescriptor(
             tensor_, dataType, layout, static_cast<int>(max_seq_length),
             static_cast<int>(batch_size), static_cast<int>(data_size),
             seq_lengths,
             static_cast<void*>(&padding_fill)) == CUDNN_STATUS_SUCCESS);
  return true;
}

CudnnFilterDescriptor::CudnnFilterDescriptor() : desc_(nullptr) {
  cudnnCreateFilterDescriptor(&desc_);
}

CudnnFilterDescriptor::~CudnnFilterDescriptor() {
  if (desc_ != nullptr) {
    cudnnDestroyFilterDescriptor(desc_);
    desc_ = nullptr;
  }
}

bool CudnnFilterDescriptor::Set(gsl::span<const int64_t> filter_dims,
                                cudnnDataType_t data_type) {
  if (!desc_)
    assert(cudnnCreateFilterDescriptor(&desc_) == CUDNN_STATUS_SUCCESS);

  int rank = gsl::narrow_cast<int>(filter_dims.size());
  InlinedVector<int> w_dims(rank);
  for (int i = 0; i < rank; i++) {
    w_dims[i] = gsl::narrow_cast<int>(filter_dims[i]);
  }

  assert(cudnnSetFilterNdDescriptor(desc_, data_type, CUDNN_TENSOR_NCHW, rank,
                                    w_dims.data()) == CUDNN_STATUS_SUCCESS);
  return true;
}

template <typename ElemType>
cudnnDataType_t CudnnTensor::GetDataType() {
  assert(false);
  // Not reachable but GCC complains
  return CUDNN_DATA_FLOAT;
}

template <>
cudnnDataType_t CudnnTensor::GetDataType<float>() {
  return CUDNN_DATA_FLOAT;
}

template <>
cudnnDataType_t CudnnTensor::GetDataType<double>() {
  return CUDNN_DATA_DOUBLE;
}

template <>
cudnnDataType_t CudnnTensor::GetDataType<half>() {
  return CUDNN_DATA_HALF;
}

template <>
cudnnDataType_t CudnnTensor::GetDataType<int8_t>() {
  return CUDNN_DATA_INT8;
}

template <>
cudnnDataType_t CudnnTensor::GetDataType<uint8_t>() {
  return CUDNN_DATA_UINT8;
}

template <>
const float Consts<float>::One = 1;

template <>
const double Consts<double>::One = 1;

template <>
const float Consts<float>::Zero = 0;

template <>
const double Consts<double>::Zero = 0;

const float Consts<half>::Zero = 0;

const float Consts<half>::One = 1;

template <>
const int8_t Consts<int8_t>::Zero = 0;

template <>
const int8_t Consts<int8_t>::One = 1;

template <>
const uint8_t Consts<uint8_t>::Zero = 0;

template <>
const uint8_t Consts<uint8_t>::One = 1;

}  // namespace Cudnn
