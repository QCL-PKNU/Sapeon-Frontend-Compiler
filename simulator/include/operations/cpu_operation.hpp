#ifndef OPERATIONS_CPU_OPERATION_HPP
#define OPERATIONS_CPU_OPERATION_HPP

#include <memory>

#include "datatype.hpp"
#include "inference_context.hpp"
#include "network/dimension.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"

class CpuOperation {
 public:
  virtual void Forward(Layer &layer, InferenceContext &ctx) = 0;
  virtual bool CheckValidOperation(Layer &layer, Dimension input_dimension) = 0;
  virtual Dimension CalculateOutputDimension(Layer &layer,
                                             Dimension input_dimension) = 0;
  virtual ~CpuOperation() {}
};

#endif  // OPERATIONS_CPU_OPERATION_HPP
