#pragma once

#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "datatype.hpp"
#include "glog/logging.h"
#include "network/network.hpp"
#include "network/tensor.hpp"

#ifdef GPU
#include <cudnn.h>
#endif

class InferenceContext {
 public:
  class CountTensor {
   public:
    CountTensor(std::shared_ptr<Tensor> tensor, const std::vector<int>& succs)
        : tensor_(tensor), count_(succs.size()) {}

    void DecreaseCount() { count_--; }

    std::shared_ptr<Tensor> GetTensor() const { return tensor_; }

    void SetTensor(std::shared_ptr<Tensor> tensor) { tensor_ = tensor; }

    bool IsUsed() const { return count_ <= 0; }

   private:
    std::shared_ptr<Tensor> tensor_;
    int count_;
  };

  using ArgMap = std::pair<int, CountTensor>;

  InferenceContext(Network& network, const Tensor& input_tensor);
  void SetLayerContext(const std::vector<int>& preds, int cur_idx,
                       size_t num_ops, dty::DataType out_dtype);
  void SetLayerContext(const std::vector<int>& preds, int cur_idx,
                       size_t num_ops, x220::DataType out_dtype);
  void SetLayerContext(const std::vector<int>& preds, int cur_idx,
                       size_t num_ops);
  void SetOutputTensor(std::shared_ptr<Tensor> tensor);
  std::shared_ptr<Tensor> InputTensor(int idx);
  std::shared_ptr<Tensor> OutputTensor();
  std::shared_ptr<Tensor> const GetLayerOutputTensor(int idx_layer) const;
  void EraseUsedTensors();
  dty::DataType out_dtype() const;
#ifdef GPU
  void SetCudnnHandle(cudnnHandle_t& handle);
#endif

 private:
  void SetTensorCache(Network& network, const Tensor& input_tensor);
  std::vector<int> CreateInputSuccessors(Network& network);
  int cur_layer_idx_;
  size_t num_next_ops_;
  size_t num_total_ops_;
  size_t num_layers_;
  std::map<int, CountTensor> tensor_cache_;
  std::vector<std::weak_ptr<Tensor>> inputs_;
  dty::DataType out_dtype_;
  dty::DataType intermediate_out_dtype_;
  std::shared_ptr<Tensor> intermediate_tensor_;
#ifdef GPU
  cudnnHandle_t handle_;
#endif
};
