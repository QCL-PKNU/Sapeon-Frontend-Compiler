#include "inference_context.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "datatype.hpp"
#include "glog/logging.h"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"

#ifdef GPU
#include <cudnn.h>
#endif

InferenceContext::InferenceContext(Network& network, const Tensor& input_tensor)
    : cur_layer_idx_(-1),
      num_next_ops_(0),
      num_total_ops_(0),
      num_layers_(network.num_layers()),
      out_dtype_(dty::DataType::FP32) {
  SetTensorCache(network, input_tensor);
}
#ifdef GPU
void InferenceContext::SetCudnnHandle(cudnnHandle_t& handle) {
  handle_ = handle;
}
#endif

void InferenceContext::SetLayerContext(const std::vector<int>& preds,
                                       int cur_idx, size_t num_ops,
                                       x220::DataType out_dtype) {
  SetLayerContext(preds, cur_idx, num_ops);
  switch (out_dtype) {
    case x220::DataType::DTY_SINT8:
      out_dtype_ = dty::DataType::SINT8;
      break;
    case x220::DataType::DTY_UINT8:
      out_dtype_ = dty::DataType::UINT8;
      break;

    case x220::DataType::DTY_SINT16:
      out_dtype_ = dty::DataType::SINT16;
    default:
      DLOG(ERROR) << "Not supported datatype for x220 inference";
      break;
  }
}

void InferenceContext::SetLayerContext(const std::vector<int>& preds,
                                       int cur_idx, size_t num_ops,
                                       dty::DataType out_dtype) {
  SetLayerContext(preds, cur_idx, num_ops);
  out_dtype_ = out_dtype_;
}

void InferenceContext::SetLayerContext(const std::vector<int>& preds,
                                       int cur_idx, size_t num_ops) {
  inputs_.clear();
  num_total_ops_ = num_ops;
  num_next_ops_ = num_ops;
  cur_layer_idx_ = cur_idx;
  if (preds.empty()) {
    auto& count_tensor = tensor_cache_.at(-1);
    inputs_.push_back(std::weak_ptr<Tensor>(count_tensor.GetTensor()));
    count_tensor.DecreaseCount();
  }
  inputs_.reserve(preds.size());
  for (int i = 0; i < preds.size(); i++) {
    auto& count_tensor = tensor_cache_.at(preds.at(i));
    inputs_.push_back(std::weak_ptr<Tensor>(count_tensor.GetTensor()));
    count_tensor.DecreaseCount();
  }
  intermediate_out_dtype_ = inputs_.begin()->lock()->dtype();
  // Use default layer output dtype (input's dtype)
  out_dtype_ = intermediate_out_dtype_;
}

void InferenceContext::SetOutputTensor(std::shared_ptr<Tensor> tensor) {
  inputs_.clear();
  if (tensor == nullptr) {
    DLOG(ERROR) << "Output tensor is nullptr";
    intermediate_tensor_ = nullptr;
    return;
  }
  if (num_next_ops_ > 1) {
    intermediate_tensor_ = tensor;
    inputs_.push_back(std::weak_ptr<Tensor>(intermediate_tensor_));
    num_next_ops_--;
  } else {
    intermediate_tensor_ = nullptr;
    tensor_cache_.at(cur_layer_idx_).SetTensor(tensor);
  }
}

std::shared_ptr<Tensor> InferenceContext::InputTensor(int idx) {
  std::shared_ptr<Tensor> input;
  if (num_next_ops_ == num_total_ops_) {
    input = inputs_.at(idx).lock();
    if (input == nullptr) {
      DLOG(ERROR) << "Layer's input tensor is not exist";
    }
    return input;
  }

  input = intermediate_tensor_;
  if (input == nullptr) {
    DLOG(ERROR) << "Layer's intermediate tensor is not exist";
  }
  return input;
}

std::shared_ptr<Tensor> InferenceContext::OutputTensor() {
  if (intermediate_tensor_ != nullptr) {
    return intermediate_tensor_;
  } else {
    return tensor_cache_.at(cur_layer_idx_).GetTensor();
  }
}

std::shared_ptr<Tensor> const InferenceContext::GetLayerOutputTensor(
    int idx_layer) const {
  auto ptr = tensor_cache_.find(idx_layer);
  if (ptr == tensor_cache_.end()) {
    DLOG(ERROR) << "Layer's output tensor is not exist";
    return nullptr;
  }
  return tensor_cache_.at(idx_layer).GetTensor();
}

void InferenceContext::EraseUsedTensors() {
  for (auto it = tensor_cache_.begin(); it != tensor_cache_.end();) {
    const auto& tensor = it->second;
    if (tensor.GetTensor() == nullptr) {
      break;
    }
    if (tensor.IsUsed()) {
      it = tensor_cache_.erase(it);
    } else {
      ++it;
    }
  }
}

void InferenceContext::SetTensorCache(Network& network,
                                      const Tensor& input_tensor) {
  for (auto& layer : network.layers()) {
    auto succs = layer.successors();
    tensor_cache_.emplace(layer.id(), CountTensor(nullptr, layer.successors()));
  }
  tensor_cache_.emplace(-1, CountTensor(std::make_shared<Tensor>(input_tensor),
                                        CreateInputSuccessors(network)));
}

std::vector<int> InferenceContext::CreateInputSuccessors(Network& network) {
  std::vector<int> input_succs;

  for (auto& layer : network.layers()) {
    auto count = std::count(layer.predecessors().begin(),
                            layer.predecessors().begin(), -1);
    if (count > 0) {
      input_succs.push_back(layer.id());
    }
  }

  if (input_succs.empty()) {
    input_succs.push_back(0);
  }

  return input_succs;
}

dty::DataType InferenceContext::out_dtype() const {
  if (num_next_ops_ > 0) {
    return intermediate_out_dtype_;
  }
  return out_dtype_;
}
