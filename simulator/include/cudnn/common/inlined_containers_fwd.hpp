#ifndef CUDNN_COMMON_INLINED_CONTAINERS_FWD_HPP
#define CUDNN_COMMON_INLINED_CONTAINERS_FWD_HPP

#include <memory>
#include <utility>
#include <vector>

namespace Cudnn {
template <typename T, size_t N = 0, typename Allocator = std::allocator<T>>
using InlinedVector = std::vector<T, Allocator>;

template <typename T, typename Allocator = std::allocator<T>>
class InlinedHashSet;

template <typename Key, typename Value,
          typename Allocator = std::allocator<std::pair<const Key, Value>>>
class InlinedHashMap;

template <typename T, typename Allocator = std::allocator<T>>
class NodeHashSet;

template <typename Key, typename Value,
          typename Allocator = std::allocator<std::pair<const Key, Value>>>
class NodeHashMap;
}  // namespace Cudnn

#endif  // CUDNN_COMMON_INLINED_CONTAINERS_FWD_HPP
