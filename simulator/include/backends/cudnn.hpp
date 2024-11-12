#ifndef BACKENDS_CUDNN_HPP
#define BACKENDS_CUDNN_HPP

#include <cudnn.h>

#include <memory>
#include <string>

#include "backends/backend.hpp"
#include "enums/error.hpp"
#include "network/tensor.hpp"
#include "tl/expected.hpp"

class CudnnBackend : public Backend {
 public:
  static std::unique_ptr<Backend> CreateBackend();
  CudnnBackend();
  ~CudnnBackend();
  tl::expected<std::shared_ptr<Tensor>, SimulatorError> Forward(
      Layer &layer, const int idx_sublayer) override;

 private:
  template <typename Type>
  std::shared_ptr<Tensor> Forward(Layer &layer, const int idx_sublayer);
  cudnnHandle_t handle_;
};

#endif  // BACKENDS_CUDNN_HPP
