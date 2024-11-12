// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CUDNN_COMMON_CUDA_COMMON_HPP
#define CUDNN_COMMON_CUDA_COMMON_HPP

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cufft.h>
#include <curand.h>
#include <cusparse.h>

#include <vector>

#include "cudnn/common/fast_divmod.hpp"
#include "gsl-lite.hpp"

namespace Cudnn {

// Type mapping for MLFloat16 to half
template <typename T>
class ToCudaType {
 public:
  typedef T MappedType;
  static MappedType FromFloat(float f) { return static_cast<T>(f); }
};

inline bool CalculateFdmStrides(gsl::span<fast_divmod> p,
                                const std::vector<int64_t>& dims) {
  int stride = 1;
  if (dims.empty() || p.size() < dims.size()) return false;
  auto rank = p.size();
  for (size_t i = 0; i < rank; i++) {
    p[rank - 1 - i] = fast_divmod(stride);
    if (i < dims.size() - 1) {
      stride *= static_cast<int>(dims[dims.size() - 1 - i]);
    }
  }
  return true;
}

class CublasMathModeSetter {
 public:
  CublasMathModeSetter(const cudaDeviceProp& prop, cublasHandle_t handle,
                       cublasMath_t mode)
      : handle_(handle) {
#if defined(CUDA_VERSION) && CUDA_VERSION < 11000
    enable_ = (mode == CUBLAS_TENSOR_OP_MATH ? prop.major >= 7 : true);
#else
    enable_ = (mode == CUBLAS_TF32_TENSOR_OP_MATH ? prop.major >= 8 : true);
#endif

    if (enable_) {
      cublasGetMathMode(handle, &mode_);
      enable_ = (mode_ != mode);
      if (enable_) {
        cublasSetMathMode(handle, mode);
      }
    }
  }

  ~CublasMathModeSetter() {
    if (enable_) {
      cublasSetMathMode(handle_, mode_);
    }
  }

 private:
  cublasHandle_t handle_;
  cublasMath_t mode_;
  bool enable_;
};

// Cublas Gemm options for half data type
class HalfGemmOptions {
 public:
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  cublasMath_t GetMathMode() const {
    if (pedantic_) {
      return CUBLAS_PEDANTIC_MATH;
    }
    return disallow_reduced_precision_reduction_ && !compute_16f_
               ? CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION
               : CUBLAS_DEFAULT_MATH;
  }

  cublasComputeType_t GetComputeType() const {
    if (compute_16f_) {
      return pedantic_ ? CUBLAS_COMPUTE_16F_PEDANTIC : CUBLAS_COMPUTE_16F;
    } else {
      return pedantic_ ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
    }
  }
#else
  cublasMath_t GetMathMode() const {
    // CublasMathModeSetter will check whether device has tensor cores later.
    return CUBLAS_TENSOR_OP_MATH;
  }

  cudaDataType GetComputeType() const {
    return compute_16f_ ? CUDA_R_16F : CUDA_R_32F;
  }
#endif

  static const HalfGemmOptions* GetInstance();

  bool IsCompute16F() const { return compute_16f_; }

  void Initialize(int value) {
    compute_16f_ = (value & 0x01) > 0;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    disallow_reduced_precision_reduction_ = (value & 0x02) > 0;
    pedantic_ = (value & 0x04) > 0;
#endif
    initialized_ = true;
  }

 private:
  // Default is FP32. Aggregate in FP16 might be faster but the cost is loss in
  // precision.
  bool compute_16f_{false};

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  // Avoid intermediate overflows in accumulation. When compute type is FP32, it
  // will not use FP16 in reduction.
  bool disallow_reduced_precision_reduction_{false};

  // For numerical robustness studies only. It is much slower.
  bool pedantic_{false};
#endif

  bool initialized_{false};

  static HalfGemmOptions instance;
};

}  // namespace Cudnn

#endif  // CUDNN_COMMON_CUDA_COMMON_HPP
