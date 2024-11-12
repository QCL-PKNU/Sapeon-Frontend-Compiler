#include "cpu/common/gemm.hpp"

#include <cmath>
#include <memory>

#include "glog/logging.h"

namespace cpu {

template <typename Type>
void GemmBin(size_t M, size_t N, size_t K, Type ALPHA, char *A, size_t lda,
             Type *B, size_t ldb, Type *C, size_t ldc) {
  size_t i, j, k;
#pragma omp parallel for simd schedule(static) default(shared)
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      char A_PART = A[i * lda + k];
      if (A_PART) {
        for (j = 0; j < N; ++j) {
          C[i * ldc + j] += B[k * ldb + j];
        }
      } else {
        for (j = 0; j < N; ++j) {
          C[i * ldc + j] -= B[k * ldb + j];
        }
      }
    }
  }
}

template <typename Type>
void Gemm(size_t TA, size_t TB, size_t M, size_t N, size_t K, Type ALPHA,
          Type *A, size_t lda, Type *B, size_t ldb, Type BETA, Type *C,
          size_t ldc) {
  GemmCpu(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

template <typename Type>
void gemm_nn(size_t M, size_t N, size_t K, Type ALPHA, Type *A, size_t lda,
             Type *B, size_t ldb, Type *C, size_t ldc) {
  size_t i, j, k;

// Print the base addresses of A, B, and C
LOG(INFO) << "Base address of A: " << static_cast<void*>(A);
LOG(INFO) << "Base address of B: " << static_cast<void*>(B);
LOG(INFO) << "Base address of C: " << static_cast<void*>(C);

// Print some more detailed information about their dimensions and strides
LOG(INFO) << "Dimensions - M: " << M << ", N: " << N << ", K: " << K;
LOG(INFO) << "Leading dimensions (strides) - lda: " << lda << ", ldb: " << ldb << ", ldc: " << ldc;

// Print the total sizes of A, B, and C in memory
size_t size_A = M * lda * sizeof(Type);
size_t size_B = K * ldb * sizeof(Type);
size_t size_C = M * ldc * sizeof(Type);
LOG(INFO) << "Memory size of A: " << size_A << " bytes";
LOG(INFO) << "Memory size of B: " << size_B << " bytes";
LOG(INFO) << "Memory size of C: " << size_C << " bytes";
// #pragma omp parallel for simd schedule(static, 64) default(shared)
  for (i = 0; i < 64; ++i) {
    for (k = 0; k < K; ++k) {
        Type A_PART = ALPHA * A[i * lda + k];
        for (j = 0; j < N; ++j) {
            C[i * ldc + j] += A_PART * B[k * ldb + j];
        }
    }
}
}

template <typename Type>
void gemm_nt(size_t M, size_t N, size_t K, Type ALPHA, Type *A, size_t lda,
             Type *B, size_t ldb, Type *C, size_t ldc) {
  size_t i, j, k;
// #pragma omp parallel for simd
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      Type sum = 0;
      for (k = 0; k < K; ++k) {
        sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
      }
      C[i * ldc + j] += sum;
    }
  }
}

template <typename Type>
void gemm_tn(size_t M, size_t N, size_t K, Type ALPHA, Type *A, size_t lda,
             Type *B, size_t ldb, Type *C, size_t ldc) {
  size_t i, j, k;
// #pragma omp parallel for simd
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      Type A_PART = ALPHA * A[k * lda + i];
      for (j = 0; j < N; ++j) {
        C[i * ldc + j] += A_PART * B[k * ldb + j];
      }
    }
  }
}

template <typename Type>
void gemm_tt(size_t M, size_t N, size_t K, Type ALPHA, Type *A, size_t lda,
             Type *B, size_t ldb, Type *C, size_t ldc) {
  size_t i, j, k;
// #pragma omp parallel for simd schedule(static) default(shared)
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      Type sum = 0;
      for (k = 0; k < K; ++k) {
        sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
      }
      C[i * ldc + j] += sum;
    }
  }
}

template <typename Type>
void GemmCpu(size_t TA, size_t TB, size_t M, size_t N, size_t K, Type ALPHA,
             Type *A, size_t lda, Type *B, size_t ldb, Type BETA, Type *C,
             size_t ldc) {
  size_t i, j;
// #pragma omp parallel for simd schedule(static) default(shared)
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      C[i * ldc + j] *= BETA;
    }
  }
  if (!TA && !TB)
    gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
  else if (TA && !TB)
    gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
  else if (!TA && TB)
    gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
  else
    gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
}

#define GEMM_DECLARE(Type)                                                     \
  template void GemmBin<Type>(size_t M, size_t N, size_t K, Type ALPHA,        \
                              char *A, size_t lda, Type *B, size_t ldb,        \
                              Type *C, size_t ldc);                            \
  template void Gemm<Type>(size_t TA, size_t TB, size_t M, size_t N, size_t K, \
                           Type ALPHA, Type * A, size_t lda, Type * B,         \
                           size_t ldb, Type BETA, Type * C, size_t ldc);       \
  template void GemmCpu<Type>(size_t TA, size_t TB, size_t M, size_t N,        \
                              size_t K, Type ALPHA, Type * A, size_t lda,      \
                              Type * B, size_t ldb, Type BETA, Type * C,       \
                              size_t ldc);

GEMM_DECLARE(double)
GEMM_DECLARE(float)
GEMM_DECLARE(int8_t)
GEMM_DECLARE(uint8_t)

}  // namespace cpu
