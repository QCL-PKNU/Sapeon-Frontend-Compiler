// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CUDNN_OPS_CONV_BASE_HPP
#define CUDNN_OPS_CONV_BASE_HPP

#include <functional>
#include <list>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cudnn/common/conv_attributes.hpp"
#include "cudnn/common/cudnn_common.hpp"
#include "cudnn/common/ort_common.hpp"
#include "cudnn/common/ort_mutex.hpp"
#include "cudnn/common/tensor_shape.hpp"

namespace Cudnn {

template <typename T>
using IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;

using ConvPadVector = ConvAttributes::ConvPadVector;

class CudnnConvolutionDescriptor final {
 public:
  CudnnConvolutionDescriptor();
  ~CudnnConvolutionDescriptor();

  bool Set(size_t rank, const gsl::span<const int64_t>& pads,
           const gsl::span<const int64_t>& strides,
           const gsl::span<const int64_t>& dilations, int groups,
           cudnnConvolutionMode_t mode, cudnnDataType_t data_type);

  operator cudnnConvolutionDescriptor_t() const { return desc_; }

 private:
  cudnnConvolutionDescriptor_t desc_;
};

template <typename T>
struct vector_hash {
  std::size_t operator()(const std::vector<T>& values) const {
    std::size_t seed = values.size();
    for (auto& val : values)
      seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};

struct tensor_shape_vector_hash {
  std::size_t operator()(const TensorShapeVector& values) const {
    std::size_t seed = values.size();
    for (auto& val : values)
      seed ^=
          std::hash<int64_t>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};

template <typename Key, typename T, typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>,
          typename ListAllocator = std::allocator<Key>>
class lru_unordered_map {
 public:
  lru_unordered_map(size_t max_size) : max_size_(max_size) {}

  void insert(const Key& key, const T& value) {
    auto it = items_.find(key);
    if (it != items_.end()) {
      it->second.value = value;
      move_to_front(it->second.lru_iterator);
      return;
    }

    while (size() + 1 > max_size_) {
      items_.erase(lru_list_.back());
      lru_list_.pop_back();
    }

    lru_list_.emplace_front(key);
    items_.emplace(key, value_type{value, lru_list_.begin()});
  }

  T& at(const Key& key) {
    auto it = items_.find(key);
    if (it == items_.end()) {
      throw std::out_of_range("There is no such key in cache");
    }
    move_to_front(it->second.lru_iterator);
    return it->second.value;
  }

  bool contains(const Key& key) const {
    return items_.find(key) != items_.end();
  }

  size_t size() const { return items_.size(); }

  void clear() {
    items_.clear();
    lru_list_.clear();
  }

 private:
  using list_type = std::list<Key, ListAllocator>;
  using iterator_type = typename list_type::iterator;
  struct value_type {
    T value;
    iterator_type lru_iterator;
  };
  using MapAllocator = std::allocator<std::pair<const Key, value_type>>;

  void move_to_front(iterator_type it) {
    lru_list_.splice(lru_list_.begin(), lru_list_, it);
  }

  size_t max_size_;
  std::unordered_map<Key, value_type, Hash, KeyEqual, MapAllocator> items_;
  list_type lru_list_;
};

// cached cudnn descriptors
constexpr size_t MAX_CACHED_ALGO_PERF_RESULTS = 10000;

template <typename AlgoPerfType>
struct CudnnConvState {
  cudnnHandle_t handle;

  // if x/w dims changed, update algo and cudnnTensors
  TensorShape last_x_dims;
  TensorShape last_w_dims;

  // these would be recomputed if x/w dims change
  TensorShape y_dims;
  TensorShapeVector y_dims_with_adjusted_pads;
  size_t workspace_bytes;
  decltype(AlgoPerfType().algo) algo = (decltype(AlgoPerfType().algo))0;
  CudnnTensor x_tensor;
  const void* x_data = nullptr;
  size_t element_size = 0;
  CudnnFilterDescriptor w_desc;
  const void* w_data = nullptr;
  CudnnTensor b_tensor;
  const void* b_data = nullptr;
  CudnnTensor y_tensor;
  TensorShape y_shape;
  void* y_data = nullptr;
  CudnnTensor z_tensor;
  const void* z_data = nullptr;
  CudnnConvolutionDescriptor conv_desc;

  struct PerfResultParams {
    decltype(AlgoPerfType().algo) algo;
    decltype(AlgoPerfType().memory) memory;
    decltype(AlgoPerfType().mathType) mathType;
  };

  lru_unordered_map<TensorShapeVector, PerfResultParams,
                    tensor_shape_vector_hash>
      cached_benchmark_results{MAX_CACHED_ALGO_PERF_RESULTS};

  // Some properties needed to support asymmetric padded Conv nodes
  bool post_slicing_required;
  TensorShapeVector slice_starts;
  TensorShapeVector slice_ends;
  TensorShapeVector slice_axes;

  // note that conv objects are shared between execution frames, and a lock is
  // needed to avoid multi-thread racing
  OrtMutex mutex;
  void* memory_for_cudnn_conv_results;
};

enum : size_t {
  AlgoSearchWorkspaceSize = 32 * 1024 * 1024,
};

cudnnStatus_t GetWorkspaceSize(
    const CudnnConvState<cudnnConvolutionFwdAlgoPerf_t>& s,
    cudnnConvolutionFwdAlgo_t algo, size_t* sz);

size_t GetMaxWorkspaceSize(
    const CudnnConvState<cudnnConvolutionFwdAlgoPerf_t>& s,
    const cudnnConvolutionFwdAlgo_t* algo, int n_algo);

}  // namespace Cudnn

#endif  // CUDNN_OPS_CONV_BASE_HPP
