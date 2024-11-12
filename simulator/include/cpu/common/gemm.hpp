#ifndef CPU_COMMON_GEMM_HPP
#define CPU_COMMON_GEMM_HPP

#include <cstddef>
#include <cstdint>

namespace cpu {

template <typename Type>
void GemmBin(size_t M, size_t N, size_t K, Type ALPHA, char *A, size_t lda,
             Type *B, size_t ldb, Type *C, size_t ldc);

template <typename Type>
void Gemm(size_t TA, size_t TB, size_t M, size_t N, size_t K, Type ALPHA,
          Type *A, size_t lda, Type *B, size_t ldb, Type BETA, Type *C,
          size_t ldc);

template <typename Type>
void GemmCpu(size_t TA, size_t TB, size_t M, size_t N, size_t K, Type ALPHA,
             Type *A, size_t lda, Type *B, size_t ldb, Type BETA, Type *C,
             size_t ldc);

}  // namespace cpu

#endif  // CPU_COMMON_GEMM_HPP
