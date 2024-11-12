#ifndef CUDNN_OPS_LAYER_NORM_IMPL_CUH
#define CUDNN_OPS_LAYER_NORM_IMPL_CUH

#include "cudnn/common/common.cuh"

namespace Cudnn {

template <typename T, typename U, typename V, bool simplified>
void HostApplyLayerNorm(const cudaDeviceProp& prop, cudaStream_t stream,
                        V* output, U* mean, U* invvar, const T* input, int n1,
                        int n2, double epsilon, const V* gamma, const V* beta);

}  // namespace Cudnn

#endif  // CUDNN_OPS_LAYER_NORM_IMPL_CUH
