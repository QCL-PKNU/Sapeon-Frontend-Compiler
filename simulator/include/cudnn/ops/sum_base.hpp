#ifndef CUDNN_OPS_SUM_BASE_HPP
#define CUDNN_OPS_SUM_BASE_HPP

#include <cudnn.h>

#include <memory>
#include <vector>

#include "network/tensor.hpp"

namespace Cudnn {

template <typename Type, cudnnDataType_t DataType>
int NoBroadcastBatchImplDispatchTarget(
    cudaStream_t stream, size_t input_count, Type **data_input,
    Type *data_output, std::vector<std::shared_ptr<Tensor>> &inputs,
    std::shared_ptr<Tensor> output);

template <typename Type, cudnnDataType_t DataType>
int BinaryImplDispatchTarget(cudaStream_t stream, size_t input_count,
                             Type **data_input, Type *data_output,
                             std::vector<std::shared_ptr<Tensor>> &inputs,
                             std::shared_ptr<Tensor> output);

template <typename Type, cudnnDataType_t DataType>
int GeneralImplDispatchTarget(cudaStream_t stream, size_t input_count,
                              Type **data_input, Type *data_output,
                              std::vector<std::shared_ptr<Tensor>> &inputs,
                              std::shared_ptr<Tensor> output);

}  // namespace Cudnn

#endif  // CUDNN_OPS_SUM_BASE_HPP
