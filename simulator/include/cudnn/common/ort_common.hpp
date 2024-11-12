#ifndef CUDNN_COMMON_ORT_COMMON_HPP
#define CUDNN_COMMON_ORT_COMMON_HPP

#include <glog/logging.h>

#include <cassert>
#include <iostream>
#include <string>

// Macros to disable the copy and/or move ctor and assignment methods
// These are usually placed in the private: declarations for a class.

#define ORT_DISALLOW_COPY(TypeName) TypeName(const TypeName&) = delete

#define ORT_DISALLOW_ASSIGNMENT(TypeName) \
  TypeName& operator=(const TypeName&) = delete

#define ORT_DISALLOW_COPY_AND_ASSIGNMENT(TypeName) \
  ORT_DISALLOW_COPY(TypeName);                     \
  ORT_DISALLOW_ASSIGNMENT(TypeName)

#define ORT_DISALLOW_MOVE(TypeName) \
  TypeName(TypeName&&) = delete;    \
  TypeName& operator=(TypeName&&) = delete

#define ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TypeName) \
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(TypeName);           \
  ORT_DISALLOW_MOVE(TypeName)

#define ORT_ENFORCE(condition, message) \
  if (!(condition)) {                   \
    std::string out_message = message;  \
    LOG(ERROR) << out_message;          \
    assert(condition);                  \
  }

namespace Cudnn {

enum class AutoPadType {
  NOTSET = 0,
  VALID = 1,
  SAME_UPPER = 2,
  SAME_LOWER = 3,
};

inline AutoPadType StringToAutoPadType(const std::string& str) {
  if (str.empty()) {
    return AutoPadType::NOTSET;
  }
  if (str == "NOTSET") {  // in onnx spec, default value is "NOTSET"
    return AutoPadType::NOTSET;
  }
  if (str == "VALID") {
    return AutoPadType::VALID;
  }
  if (str == "SAME_UPPER") {
    return AutoPadType::SAME_UPPER;
  }
  if (str == "SAME_LOWER") {
    return AutoPadType::SAME_LOWER;
  }
  ORT_ENFORCE(false, "Unknown AutoPadType String");
  return AutoPadType::NOTSET;
}

// helper function

inline bool ComputePad(const int64_t in_dim, const int64_t stride,
                       const int64_t kernel, const int64_t dilation,
                       AutoPadType pad_type, int64_t& pad_head,
                       int64_t& pad_tail,
                       bool force_symmetric_auto_padding = false) {
  switch (pad_type) {
    case AutoPadType::NOTSET:
      break;
    case AutoPadType::VALID: {
      pad_head = 0;
      pad_tail = 0;
    } break;
    case AutoPadType::SAME_UPPER:
    case AutoPadType::SAME_LOWER: {
      if (1 != dilation)
        LOG(ERROR) << "Dilation not supported for AutoPadType::SAME_UPPER or "
                      "AutoPadType::SAME_LOWER.";
      return false;

      // The ONNX spec says if `auto_pad` attribute is set, pad until the
      // `legacy_target_size` is `ceil (in_dim / stride)`. The following line of
      // code is essentially just that and is retained as is
      int64_t legacy_target_size = (in_dim + stride - 1) / stride;
      int64_t pad_needed = (legacy_target_size - 1) * stride + kernel - in_dim;
      // make sure padding is symmetric
      if (force_symmetric_auto_padding) {
        // Inlining math::roundUpPow2() from util/math.h to avoid bringing in
        // the transitive dependencies.
        pad_needed = (pad_needed + 1) & ~1;
      }

      if (pad_type == AutoPadType::SAME_LOWER)
        pad_head = (pad_needed + 1) / 2;
      else
        pad_head = pad_needed / 2;

      pad_tail = pad_needed - pad_head;
    } break;
    default:
      LOG(ERROR) << "ComputePad: pad type not supported.";
      return false;
  }

  return true;
}

constexpr inline int64_t ComputeOutputShape(
    const int64_t in_dim, const int64_t stride, const int64_t kernel,
    const int64_t dilation, const int64_t pad_head, const int64_t pad_tail) {
  const int64_t dkernel = dilation * (kernel - 1) + 1;
  return static_cast<int64_t>(
      static_cast<double>(in_dim + pad_head + pad_tail - dkernel) / stride + 1);
}

inline bool ComputePadAndOutputShape(
    const int64_t in_dim, const int64_t stride, const int64_t kernel,
    const int64_t dilation, AutoPadType pad_type, int64_t& pad_head,
    int64_t& pad_tail, int64_t& out_dim,
    bool force_symmetric_auto_padding = false) {
  assert(ComputePad(in_dim, stride, kernel, dilation, pad_type, pad_head,
                    pad_tail, force_symmetric_auto_padding));
  out_dim =
      ComputeOutputShape(in_dim, stride, kernel, dilation, pad_head, pad_tail);
  return true;
}

}  // namespace Cudnn

#endif  // CUDNN_COMMON_ORT_COMMON_HPP
