// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cudnn/common/cuda_common.hpp"

namespace Cudnn {

// The environment variable is for testing purpose only, and it might be removed
// in the future. The value is an integer, and its bits have the following
// meaning:
//   0x01 - aggregate in fp16
//   0x02 - disallow reduced precision reduction. No effect when aggregate in
//   fp16. 0x04 - pedantic
constexpr const char* kCudaGemmOptions = "ORT_CUDA_GEMM_OPTIONS";

// Initialize the singleton instance
HalfGemmOptions HalfGemmOptions::instance;

const HalfGemmOptions* HalfGemmOptions::GetInstance() {
  if (!instance.initialized_) {
    // We do not use critical section here since it is fine to initialize
    // multiple times by different threads.
    // int value = ParseEnvironmentVariableWithDefault<int>(kCudaGemmOptions,
    // 0); instance.Initialize(value);
    instance.Initialize(0);
  }

  return &instance;
}

}  // namespace Cudnn
