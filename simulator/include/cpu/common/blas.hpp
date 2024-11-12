#ifndef CPU_COMMON_BLAS_HPP
#define CPU_COMMON_BLAS_HPP

#include <cstddef>
#include <cstdint>

namespace cpu {

template <typename Type>
void CopyCpu(size_t N, Type *X, size_t INCX, Type *Y, size_t INCY);

template <typename Type>
void ScaleBias(Type *output, Type *scales, size_t batch, size_t n, size_t size);

template <typename Type>
void NormalizeCpu(Type *x, Type *mean, Type *variance, size_t batch,
                  size_t filters, size_t spatial, float epsilon);

template <typename Type>
void AddBias(Type *output, Type *biases, size_t batch, size_t n, size_t size);

}  // namespace cpu

#endif  // CPU_COMMON_BLAS_HPP
