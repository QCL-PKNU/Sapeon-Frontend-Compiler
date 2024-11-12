#include "cpu/common/blas.hpp"

#include <cmath>
#include <iostream>

namespace cpu {

template <typename Type>
void CopyCpu(size_t N, Type *X, size_t INCX, Type *Y, size_t INCY) {
  size_t i;
#pragma omp parallel for simd schedule(static) default(shared)
  for (i = 0; i < N; ++i) Y[i * INCY] = X[i * INCX];
}

template <typename Type>
void ScaleBias(Type *output, Type *scales, size_t batch, size_t n,
               size_t size) {
  size_t i, j, b;
#pragma omp parallel for simd schedule(static) default(shared) collapse(3)
  for (b = 0; b < batch; ++b) {
    for (i = 0; i < n; ++i) {
      for (j = 0; j < size; ++j) {
        output[(b * n + i) * size + j] *= scales[i];
      }
    }
  }
}

template <typename Type>
void AddBias(Type *output, Type *biases, size_t batch, size_t n, size_t size) {
  size_t i, j, b;
#pragma omp parallel for simd schedule(static) default(shared) collapse(3)
  for (b = 0; b < batch; ++b) {
    for (i = 0; i < n; ++i) {
      for (j = 0; j < size; ++j) {
        // if(output[(b*n + i) * size + j] != 0.0)
        output[(b * n + i) * size + j] += biases[i];
      }
    }
  }
}

template <typename Type>
void NormalizeCpu(Type *x, Type *mean, Type *variance, size_t batch,
                  size_t filters, size_t spatial, float epsilon) {
  size_t b, f, i;
#pragma omp parallel for simd schedule(static) default(shared) collapse(3)
  for (b = 0; b < batch; ++b) {
    for (f = 0; f < filters; ++f) {
      for (i = 0; i < spatial; ++i) {
        size_t index = b * filters * spatial + f * spatial + i;
        x[index] = (x[index] - mean[f]) / (sqrt(variance[f] + epsilon));
      }
    }
  }
}

/*
template <>
void NormalizeCpu<int8_t>(int8_t *x, int8_t *mean, int8_t *variance,
                           size_t batch, size_t filters, size_t spatial,
                           float epsilon) {
  size_t b, f, i;
  for (b = 0; b < batch; ++b) {
    for (f = 0; f < filters; ++f) {
      for (i = 0; i < spatial; ++i) {
        size_t index = b * filters * spatial + f * spatial + i;
        x[index] = (int8_t)((double)(x[index] - mean[f]) /
                            (double)(sqrt(variance[f] + epsilon)));
      }
    }
  }
}
*/

#define DECLARE_NORMAL_INT(Type)                                        \
  template <>                                                           \
  void NormalizeCpu<Type>(Type * x, Type * mean, Type * variance,       \
                          size_t batch, size_t filters, size_t spatial, \
                          float epsilon) {                              \
    size_t b, f, i;                                                     \
    for (b = 0; b < batch; ++b) {                                       \
      for (f = 0; f < filters; ++f) {                                   \
        for (i = 0; i < spatial; ++i) {                                 \
          size_t index = b * filters * spatial + f * spatial + i;       \
          x[index] = (Type)((double)(x[index] - mean[f]) /              \
                            (double)(sqrt(variance[f] + epsilon)));     \
        }                                                               \
      }                                                                 \
    }                                                                   \
  }

template void ScaleBias<float>(float *output, float *scales, size_t batch,
                               size_t n, size_t size);
template void ScaleBias<double>(double *output, double *scales, size_t batch,
                                size_t n, size_t size);
template void ScaleBias<int8_t>(int8_t *output, int8_t *scales, size_t batch,
                                size_t n, size_t size);
template void ScaleBias<uint8_t>(uint8_t *output, uint8_t *scales, size_t batch,
                                 size_t n, size_t size);

template void AddBias<float>(float *output, float *scales, size_t batch,
                             size_t n, size_t size);
template void AddBias<double>(double *output, double *scales, size_t batch,
                              size_t n, size_t size);
template void AddBias<int8_t>(int8_t *output, int8_t *scales, size_t batch,
                              size_t n, size_t size);
template void AddBias<uint8_t>(uint8_t *output, uint8_t *scales, size_t batch,
                               size_t n, size_t size);

template void NormalizeCpu<float>(float *x, float *mean, float *variance,
                                  size_t batch, size_t filters, size_t spatial,
                                  float epsilon);
template void NormalizeCpu<double>(double *x, double *mean, double *variance,
                                   size_t batch, size_t filters, size_t spatial,
                                   float epsilon);
template void NormalizeCpu<uint8_t>(uint8_t *x, uint8_t *mean,
                                    uint8_t *variance, size_t batch,
                                    size_t filters, size_t spatial,
                                    float epsilon);
template void NormalizeCpu<int8_t>(int8_t *x, int8_t *mean, int8_t *variance,
                                   size_t batch, size_t filters, size_t spatial,
                                   float epsilon);

}  // namespace cpu
