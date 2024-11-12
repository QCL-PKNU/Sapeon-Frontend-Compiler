#ifndef CUDNN_COMMON_ATOMIC_COMMON_CUH
#define CUDNN_COMMON_ATOMIC_COMMON_CUH

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace Cudnn {

__device__ __forceinline__ void atomic_add(float* address, float value) {
  atomicAdd(address, value);
}

__device__ __forceinline__ void atomic_add(double* address, double value) {
#if __CUDA_ARCH__ < 600
  unsigned long long* raw_address =
      reinterpret_cast<unsigned long long*>(address);
  unsigned long long raw_old_value = 0ULL;
  unsigned long long raw_new_value = 0ULL;
  unsigned long long seen_old_value = 0ULL;
  double* const p_old_value = reinterpret_cast<double*>(&raw_old_value);
  double* const p_new_value = reinterpret_cast<double*>(&raw_new_value);
  do {
    *p_old_value = *address;
    *p_new_value = *address + value;
    seen_old_value = atomicCAS(raw_address, raw_old_value, raw_new_value);
  } while (seen_old_value != raw_old_value);
#else
  atomicAdd(address, value);
#endif
}

//
// ref:
// https://github.com/pytorch/pytorch/blob/master/aten/src/THC/THCAtomics.cuh
//
__device__ __forceinline__ void atomic_add(half* address, half value) {
#if __CUDA_ARCH__ < 700
  unsigned int* base_address =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *base_address;
  unsigned int assumed;
  unsigned short x;

  do {
    assumed = old;
    x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    x = __half_as_short(
        __float2half(__half2float(*reinterpret_cast<const __half*>(&x)) +
                     __half2float(value)));
    old = (size_t)address & 2 ? (old & 0xffff) | (x << 16)
                              : (old & 0xffff0000) | x;
    old = atomicCAS(base_address, assumed, old);
  } while (assumed != old);
#else
  atomicAdd(address, value);
#endif
}

}  // namespace Cudnn

#endif  // CUDNN_COMMON_ATOMIC_COMMON_CUH