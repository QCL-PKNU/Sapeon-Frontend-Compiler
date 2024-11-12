// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CUDNN_COMMON_INLINED_CONTAINERS_HPP
#define CUDNN_COMMON_INLINED_CONTAINERS_HPP

#include <cmath>
#include <functional>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "cudnn/common/inlined_containers_fwd.hpp"

namespace Cudnn {

template <typename T, typename Allocator>
class InlinedHashSet
    : public std::unordered_set<T, std::hash<T>, std::equal_to<T>, Allocator> {
  using Base = std::unordered_set<T, std::hash<T>, std::equal_to<T>, Allocator>;

 public:
  using Base::Base;
};

template <typename Key, typename Value, typename Allocator>
class InlinedHashMap
    : public std::unordered_map<Key, Value, std::hash<Key>, std::equal_to<Key>,
                                Allocator> {
  using Base = std::unordered_map<Key, Value, std::hash<Key>,
                                  std::equal_to<Key>, Allocator>;

 public:
  using Base::Base;
};

// Use this hash set/map where pointer stability is required, otherwise use
// InlinedHashSet and InlinedHashMap
// This does not allocate a dummy 'end' node on default construction.
// Use reserve() when the number of elements is known.
template <typename T, typename Allocator>
class NodeHashSet
    : public std::unordered_set<T, std::hash<T>, std::equal_to<T>, Allocator> {
  using Base = std::unordered_set<T, std::hash<T>, std::equal_to<T>, Allocator>;

 public:
  using Base::Base;
};

template <typename Key, typename Value, typename Allocator>
class NodeHashMap : public std::unordered_map<Key, Value, std::hash<Key>,
                                              std::equal_to<Key>, Allocator> {
  using Base = std::unordered_map<Key, Value, std::hash<Key>,
                                  std::equal_to<Key>, Allocator>;

 public:
  using Base::Base;
};

}  // namespace Cudnn

#endif  // CUDNN_COMMON_INLINED_CONTAINERS_HPP
