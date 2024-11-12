#ifndef OPERATIONS_CUDNN_OPERATION_HPP
#define OPERATIONS_CUDNN_OPERATION_HPP

#include <cudnn.h>

#include <memory>

#include "network/layer.hpp"
#include "network/tensor.hpp"

template <typename Type>
class CudnnOperation {
 public:
  virtual std::shared_ptr<Tensor> Forward(cudnnHandle_t &handle,
                                          Layer &layer) = 0;
  virtual ~CudnnOperation() {}
};

#endif  // OPERATIONS_CUDNN_OPERATION_HPP
