//
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for
// full license information.
//

// Make generic operators for floating point types
/* This file contains:
   Generalized library calls
   kernels to be called for not supported data type
*/
// NV_TODO: optimize speed -- pass things needed in, optimize kernel speed, add
// half2 NV_TODO: investigate cub support for half

#ifndef CUDNN_COMMON_FPGENERIC_HPP
#define CUDNN_COMMON_FPGENERIC_HPP

#include "cudnn/common/cuda_common.hpp"

using namespace Cudnn;

static float halfToFloat(u_int16_t h) {
  union {
    float f;
    unsigned int i;
  } value;

  int s = (h >> 15) & 0x00000001;  // sign
  int e = (h >> 10) & 0x0000001f;  // exponent
  int f = h & 0x000003ff;          // fraction

  // need to handle 7c00 INF and fc00 -INF?
  if (e == 0) {
    // need to handle +-0 case f==0 or f=0x8000?
    if (f == 0)  // Plus or minus zero
      value.i = s << 31;
    else {  // Denormalized number -- renormalize it
      while (!(f & 0x00000400)) {
        f <<= 1;
        e -= 1;
      }
      e += 1;
      f &= ~0x00000400;
    }
  } else if (e == 31) {
    if (f == 0)  // Inf
      value.i = (s << 31) | 0x7f800000;
    else  // NaN
      value.i = (s << 31) | 0x7f800000 | (f << 13);
  }

  e = e + (127 - 15);
  f = f << 13;

  value.i = ((s << 31) | (e << 23) | f);

  return value.f;
}

static uint16_t floatToHalf(float input) {
  union {
    float f;
    unsigned int i;
  } value = {input};

  uint16_t h;

  unsigned int i = value.i;

  register int s = (i >> 16) & 0x00008000;                 // sign
  register int e = ((i >> 23) & 0x000000ff) - (127 - 15);  // exponent
  register int f = i & 0x007fffff;                         // fraction

  // need to handle NaNs and Inf?
  if (e <= 0) {
    if (e < -10) {
      if (s)  // handle -0.0
        h = 0x8000;
      else
        h = 0;
    }

    f = (f | 0x00800000) >> (1 - e);
    h = s | (f >> 13);
  } else if (e == 0xff - (127 - 15)) {
    if (f == 0)  // Inf
      h = s | 0x7c00;
    else {  // NAN
      f >>= 13;
      h = s | 0x7c00 | f | (f == 0);
    }
  } else {
    if (e > 30)  // Overflow
      h = s | 0x7c00;

    h = s | (e << 10) | (f >> 13);
  }
  return h;
}

// Generalize library calls to be use in template functions
inline cublasStatus_t cublasGemmHelper(cublasHandle_t handle,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb, int m, int n,
                                       int k, const float* alpha,
                                       const float* A, int lda, const float* B,
                                       int ldb, const float* beta, float* C,
                                       int ldc, const cudaDeviceProp& prop) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  // TF32 uses 10 bit mantissa which has sufficient margin of precision for
  // most use cases. It gets 8x throughput than FP32 in A100. It can be
  // overrided by setting environment variable NVIDIA_TF32_OVERRIDE = 0 to
  // disable TF32 - CUBLAS_TF32_TENSOR_OP_MATH
  Cudnn::CublasMathModeSetter math_mode_setter(prop, handle,
                                               CUBLAS_DEFAULT_MATH);
#else
  ORT_UNUSED_PARAMETER(prop);
#endif

  return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}

inline cublasStatus_t cublasGemmHelper(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const double* alpha, const double* A, int lda,
    const double* B, int ldb, const double* beta, double* C, int ldc,
    const cudaDeviceProp& /*prop*/) {
  return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}

inline cublasStatus_t cublasGemmHelper(cublasHandle_t handle,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb, int m, int n,
                                       int k, const half* alpha, const half* A,
                                       int lda, const half* B, int ldb,
                                       const half* beta, half* C, int ldc,
                                       const cudaDeviceProp& prop) {
  const HalfGemmOptions* half_options = HalfGemmOptions::GetInstance();
  Cudnn::CublasMathModeSetter math_mode_setter(prop, handle,
                                               half_options->GetMathMode());
  if (half_options->IsCompute16F()) {
    return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, CUDA_R_16F,
                        lda, B, CUDA_R_16F, ldb, beta, C, CUDA_R_16F, ldc,
                        half_options->GetComputeType(), CUBLAS_GEMM_DEFAULT);
  } else {
    // The alpha and beta shall have same precision as compute type.
    float h_a = halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
    float h_b = halfToFloat(*reinterpret_cast<const uint16_t*>(beta));
    return cublasGemmEx(handle, transa, transb, m, n, k, &h_a, A, CUDA_R_16F,
                        lda, B, CUDA_R_16F, ldb, &h_b, C, CUDA_R_16F, ldc,
                        half_options->GetComputeType(), CUBLAS_GEMM_DEFAULT);
  }
}

inline cublasStatus_t cublasGemmHelper(cublasHandle_t handle,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb, int m, int n,
                                       int k, const float* alpha, const half* A,
                                       int lda, const half* B, int ldb,
                                       const float* beta, half* C, int ldc,
                                       const cudaDeviceProp& prop) {
  const HalfGemmOptions* half_options = HalfGemmOptions::GetInstance();
  Cudnn::CublasMathModeSetter math_mode_setter(prop, handle,
                                               half_options->GetMathMode());
  if (half_options->IsCompute16F()) {
    // The alpha and beta shall have same precision as compute type.
    uint16_t h_a = floatToHalf(*alpha);
    uint16_t h_b = floatToHalf(*beta);
    return cublasGemmEx(handle, transa, transb, m, n, k, &h_a, A, CUDA_R_16F,
                        lda, B, CUDA_R_16F, ldb, &h_b, C, CUDA_R_16F, ldc,
                        half_options->GetComputeType(), CUBLAS_GEMM_DEFAULT);
  } else {
    return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, CUDA_R_16F,
                        lda, B, CUDA_R_16F, ldb, beta, C, CUDA_R_16F, ldc,
                        half_options->GetComputeType(), CUBLAS_GEMM_DEFAULT);
  }
}

// batched gemm --- CUBLAS_TF32_TENSOR_OP_MATH
inline cublasStatus_t cublasGemmBatchedHelper(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* Aarray[], int lda,
    const float* Barray[], int ldb, const float* beta, float* Carray[], int ldc,
    int batch_count, const cudaDeviceProp& prop) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  Cudnn::CublasMathModeSetter math_mode_setter(prop, handle,
                                               CUBLAS_DEFAULT_MATH);
#else
  ORT_UNUSED_PARAMETER(prop);
#endif

  return cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                            Barray, ldb, beta, Carray, ldc, batch_count);
}

inline cublasStatus_t cublasGemmBatchedHelper(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const double* alpha, const double* Aarray[], int lda,
    const double* Barray[], int ldb, const double* beta, double* Carray[],
    int ldc, int batch_count, const cudaDeviceProp& /*prop*/) {
  return cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                            Barray, ldb, beta, Carray, ldc, batch_count);
}

inline cublasStatus_t cublasGemmBatchedHelper(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const half* alpha, const half* Aarray[], int lda,
    const half* Barray[], int ldb, const half* beta, half* Carray[], int ldc,
    int batch_count, const cudaDeviceProp& prop) {
  const HalfGemmOptions* half_options = HalfGemmOptions::GetInstance();
  Cudnn::CublasMathModeSetter math_mode_setter(prop, handle,
                                               half_options->GetMathMode());
  if (half_options->IsCompute16F()) {
    return cublasGemmBatchedEx(
        handle, transa, transb, m, n, k, alpha, (const void**)Aarray,
        CUDA_R_16F, lda, (const void**)Barray, CUDA_R_16F, ldb, beta,
        (void**)Carray, CUDA_R_16F, ldc, batch_count,
        half_options->GetComputeType(), CUBLAS_GEMM_DEFAULT);
  } else {
    float h_a = halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
    float h_b = halfToFloat(*reinterpret_cast<const uint16_t*>(beta));
    return cublasGemmBatchedEx(
        handle, transa, transb, m, n, k, &h_a, (const void**)Aarray, CUDA_R_16F,
        lda, (const void**)Barray, CUDA_R_16F, ldb, &h_b, (void**)Carray,
        CUDA_R_16F, ldc, batch_count, half_options->GetComputeType(),
        CUBLAS_GEMM_DEFAULT);
  }
}

// strided batched gemm
inline cublasStatus_t cublasGemmStridedBatchedHelper(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* A, int lda,
    long long int strideA, const float* B, int ldb, long long int strideB,
    const float* beta, float* C, int ldc, long long int strideC,
    int batch_count, const cudaDeviceProp& prop) {
  return cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A,
                                   lda, strideA, B, ldb, strideB, beta, C, ldc,
                                   strideC, batch_count);
}

inline cublasStatus_t cublasGemmStridedBatchedHelper(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const double* alpha, const double* A, int lda,
    long long int strideA, const double* B, int ldb, long long int strideB,
    const double* beta, double* C, int ldc, long long int strideC,
    int batch_count, const cudaDeviceProp& /*prop*/) {
  return cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A,
                                   lda, strideA, B, ldb, strideB, beta, C, ldc,
                                   strideC, batch_count);
}

inline cublasStatus_t cublasGemmStridedBatchedHelper(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const __half* alpha, const __half* A, int lda,
    long long int strideA, const __half* B, int ldb, long long int strideB,
    const __half* beta, __half* C, int ldc, long long int strideC,
    int batch_count, const cudaDeviceProp& prop) {
  const HalfGemmOptions* half_options = HalfGemmOptions::GetInstance();
  Cudnn::CublasMathModeSetter math_mode_setter(prop, handle,
                                               half_options->GetMathMode());
  if (half_options->IsCompute16F()) {
    return cublasGemmStridedBatchedEx(
        handle, transa, transb, m, n, k, alpha, A, CUDA_R_16F, lda, strideA, B,
        CUDA_R_16F, ldb, strideB, beta, C, CUDA_R_16F, ldc, strideC,
        batch_count, half_options->GetComputeType(), CUBLAS_GEMM_DEFAULT);
  } else {
    // The alpha and beta shall have same precision as compute type.
    float h_a = halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
    float h_b = halfToFloat(*reinterpret_cast<const uint16_t*>(beta));
    return cublasGemmStridedBatchedEx(
        handle, transa, transb, m, n, k, &h_a, A, CUDA_R_16F, lda, strideA, B,
        CUDA_R_16F, ldb, strideB, &h_b, C, CUDA_R_16F, ldc, strideC,
        batch_count, half_options->GetComputeType(), CUBLAS_GEMM_DEFAULT);
  }
}

inline cublasStatus_t cublasGemmStridedBatchedHelper(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float* alpha, const __half* A, int lda,
    long long int strideA, const __half* B, int ldb, long long int strideB,
    const float* beta, __half* C, int ldc, long long int strideC,
    int batch_count, const cudaDeviceProp& prop) {
  const HalfGemmOptions* half_options = HalfGemmOptions::GetInstance();
  Cudnn::CublasMathModeSetter math_mode_setter(prop, handle,
                                               half_options->GetMathMode());
  if (half_options->IsCompute16F()) {
    // The alpha and beta shall have same precision as compute type.
    uint16_t h_a = floatToHalf(*alpha);
    uint16_t h_b = floatToHalf(*beta);
    return cublasGemmStridedBatchedEx(
        handle, transa, transb, m, n, k, &h_a, A, CUDA_R_16F, lda, strideA, B,
        CUDA_R_16F, ldb, strideB, &h_b, C, CUDA_R_16F, ldc, strideC,
        batch_count, half_options->GetComputeType(), CUBLAS_GEMM_DEFAULT);
  } else {
    return cublasGemmStridedBatchedEx(
        handle, transa, transb, m, n, k, alpha, A, CUDA_R_16F, lda, strideA, B,
        CUDA_R_16F, ldb, strideB, beta, C, CUDA_R_16F, ldc, strideC,
        batch_count, half_options->GetComputeType(), CUBLAS_GEMM_DEFAULT);
  }
}

// transpose using geam
inline cublasStatus_t cublasTransposeHelper(
    cudaStream_t, cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, const float* alpha, const float* A,
    int lda, const float* beta, const float* B, int ldb, float* C, int ldc) {
  return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb,
                     C, ldc);
}

inline cublasStatus_t cublasTransposeHelper(cudaStream_t, cublasHandle_t handle,
                                            cublasOperation_t transa,
                                            cublasOperation_t transb, int m,
                                            int n, const double* alpha,
                                            const double* A, int lda,
                                            const double* beta, const double* B,
                                            int ldb, double* C, int ldc) {
  return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb,
                     C, ldc);
}

cublasStatus_t cublasTransposeHelper(cudaStream_t, cublasHandle_t,
                                     cublasOperation_t, cublasOperation_t,
                                     int m, int n, const half*, const half* A,
                                     int, const half*, const half*, int,
                                     half* C, int);

// copy
inline cublasStatus_t cublasCopyHelper(cudaStream_t, cublasHandle_t handle,
                                       int n, const float* x, int incx,
                                       float* y, int incy) {
  return cublasScopy(handle, n, x, incx, y, incy);
}

inline cublasStatus_t cublasCopyHelper(cudaStream_t, cublasHandle_t handle,
                                       int n, const double* x, int incx,
                                       double* y, int incy) {
  return cublasDcopy(handle, n, x, incx, y, incy);
}

cublasStatus_t cublasCopyHelper(cudaStream_t stream, cublasHandle_t handle,
                                int n, const half* x, int incx, half* y,
                                int incy);

#endif  // CUDNN_COMMON_FPGENERIC_HPP
