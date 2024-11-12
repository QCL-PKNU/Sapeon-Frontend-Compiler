#include <cudnn.h>
#include <glog/logging.h>

#include <memory>
using std::shared_ptr;
#include <vector>
using std::vector;

#include "cudnn/common/binary_elementwise.hpp"
#include "cudnn/common/cuda_common.hpp"
#include "cudnn/common/variadic_elementwise.hpp"
#include "network/tensor.hpp"

namespace Cudnn {

template <typename Type, cudnnDataType_t DataType>
int NoBroadcastBatchImplDispatchTarget(cudaStream_t stream, size_t input_count,
                                       Type **data_input, Type *data_output,
                                       vector<shared_ptr<Tensor>> &inputs,
                                       std::shared_ptr<Tensor> output) {
  using CudaT = typename ToCudaType<Type>::MappedType;
  vector<vector<int64_t>> in_vectors;
  vector<int64_t> out_vector = output->dimension().dims();
  for (size_t i = 0; i < input_count; i++) {
    in_vectors.push_back(inputs[i]->dimension().dims());
  }

  assert(input_count > 1);
  size_t index =
      std::min(input_count, static_cast<size_t>(k_max_input_batch_size));

  InputBatchArray<CudaT> input_data_batch{static_cast<int32_t>(index)};
  for (size_t i = 0; i < index; ++i) {
    input_data_batch[static_cast<int32_t>(i)] =
        reinterpret_cast<const CudaT *>(data_input);
  }

  CudaT *output_data = reinterpret_cast<CudaT *>(data_output);

  Impl_NoBroadcastInputBatch<CudaT, variadic_elementwise_ops::Sum>(
      stream, input_data_batch, output_data, output->dimension().size());

  while (index < input_count) {
    size_t left_count = input_count - index + 1;
    size_t batch =
        std::min(left_count, static_cast<size_t>(k_max_input_batch_size));
    // Special case for 2 inputs left.
    if (batch == 2) {
      size_t count = output->dimension().size();

      int32_t output_rank_or_simple_broadcast;
      fast_divmod fdm_H, fdm_C;
      TArray<int64_t> lhs_padded_strides, rhs_padded_strides;
      TArray<fast_divmod> fdm_output_strides;

      BinaryElementwiseBroadcastPrepare(
          out_vector, in_vectors[input_count - 1], out_vector,
          output_rank_or_simple_broadcast, fdm_H, fdm_C, lhs_padded_strides,
          rhs_padded_strides, fdm_output_strides);

      Impl_General<CudaT, variadic_elementwise_ops::Sum>(
          stream, output_rank_or_simple_broadcast, &lhs_padded_strides,
          reinterpret_cast<const CudaT *>(data_output), &rhs_padded_strides,
          reinterpret_cast<const CudaT *>(data_input[input_count - 1]),
          &fdm_output_strides, fdm_H, fdm_C,
          reinterpret_cast<CudaT *>(data_output), output->dimension().size());

      // Must be the last.
      break;
    }

    InputBatchArray<CudaT> left_input_data_batch{static_cast<int32_t>(batch)};
    left_input_data_batch[0] = reinterpret_cast<const CudaT *>(data_output);
    for (size_t i = 1; i < batch; ++i) {
      left_input_data_batch[static_cast<int32_t>(i)] =
          reinterpret_cast<const CudaT *>(data_input[index]);
      index++;
    }

    Impl_NoBroadcastInputBatch<CudaT, variadic_elementwise_ops::Sum>(
        stream, left_input_data_batch, output_data, output->dimension().size());
  }

  return 1;
}

template <typename Type, cudnnDataType_t DataType>
int BinaryImplDispatchTarget(cudaStream_t stream, size_t input_count,
                             Type **data_input, Type *data_output,
                             vector<shared_ptr<Tensor>> &inputs,
                             std::shared_ptr<Tensor> output) {
  vector<vector<int64_t>> in_vectors;
  vector<int64_t> out_vector = output->dimension().dims();
  for (size_t i = 0; i < input_count; i++) {
    in_vectors.push_back(inputs[i]->dimension().dims());
  }
  using CudaT = typename ToCudaType<Type>::MappedType;

  int32_t output_rank_or_simple_broadcast;
  fast_divmod fdm_H, fdm_C;
  TArray<int64_t> lhs_padded_strides, rhs_padded_strides;
  TArray<fast_divmod> fdm_output_strides;

  BinaryElementwiseBroadcastPrepare(
      in_vectors[0], in_vectors[1], out_vector, output_rank_or_simple_broadcast,
      fdm_H, fdm_C, lhs_padded_strides, rhs_padded_strides, fdm_output_strides);

  Impl_General<CudaT, variadic_elementwise_ops::Sum>(
      stream, output_rank_or_simple_broadcast, &lhs_padded_strides,
      reinterpret_cast<const CudaT *>(data_input[0]), &rhs_padded_strides,
      reinterpret_cast<const CudaT *>(data_input[1]), &fdm_output_strides,
      fdm_H, fdm_C, reinterpret_cast<CudaT *>(data_output),
      output->dimension().size());

  return 1;
}

template <typename Type, cudnnDataType_t DataType>
int GeneralImplDispatchTarget(cudaStream_t stream, size_t input_count,
                              Type **data_input, Type *data_output,
                              vector<shared_ptr<Tensor>> &inputs,
                              shared_ptr<Tensor> output) {
  assert(input_count > 1);

  vector<vector<int64_t>> in_vectors;
  vector<int64_t> out_vector = output->dimension().dims();
  for (size_t i = 0; i < input_count; i++) {
    in_vectors.push_back(inputs[i]->dimension().dims());
  }

  using CudaT = typename ToCudaType<Type>::MappedType;

  // If there is any input having the same shape with output, we don't need the
  // memset.
  size_t index_of_same_shape = 0;
  for (; index_of_same_shape < input_count; index_of_same_shape++) {
    if (in_vectors[index_of_same_shape] == out_vector) {
      break;
    }
  }

  int32_t output_rank_or_simple_broadcast;
  fast_divmod fdm_H, fdm_C;
  TArray<int64_t> lhs_padded_strides, rhs_padded_strides;
  TArray<fast_divmod> fdm_output_strides;

  // No input has same shape of output, memset the output, and add the 1st input
  // as initialization.
  if (index_of_same_shape == input_count) {
    cudaMemsetAsync(data_output, 0, output->size(), stream);
    BinaryElementwiseBroadcastPrepare(out_vector, in_vectors[0], out_vector,
                                      output_rank_or_simple_broadcast, fdm_H,
                                      fdm_C, lhs_padded_strides,
                                      rhs_padded_strides, fdm_output_strides);
    Impl_Add(stream, output_rank_or_simple_broadcast, &lhs_padded_strides,
             reinterpret_cast<const CudaT *>(data_output), &rhs_padded_strides,
             reinterpret_cast<const CudaT *>(data_input[0]),
             &fdm_output_strides, fdm_H, fdm_C,
             reinterpret_cast<CudaT *>(data_output),
             output->dimension().size());
  } else {
    // First operation is between input[0] and input[index_of_same_shape] if
    // index_of_same_shape is not 0.
    size_t index = index_of_same_shape == 0 ? 1 : 0;
    BinaryElementwiseBroadcastPrepare(
        in_vectors[index_of_same_shape], in_vectors[index], out_vector,
        output_rank_or_simple_broadcast, fdm_H, fdm_C, lhs_padded_strides,
        rhs_padded_strides, fdm_output_strides);
    Impl_General<CudaT, variadic_elementwise_ops::Sum>(
        stream, output_rank_or_simple_broadcast, &lhs_padded_strides,
        reinterpret_cast<const CudaT *>(data_input[index_of_same_shape]),
        &rhs_padded_strides, reinterpret_cast<const CudaT *>(data_input[index]),
        &fdm_output_strides, fdm_H, fdm_C,
        reinterpret_cast<CudaT *>(data_output), output->dimension().size());
  }

  for (size_t index = 1; index < input_count; index++) {
    // If index_of_same_shape is 0, we already handle the 1st and 2nd inputs.
    if (index == index_of_same_shape ||
        (index_of_same_shape == 0 && index == 1)) {
      continue;
    }

    BinaryElementwiseBroadcastPrepare(out_vector, in_vectors[index], out_vector,
                                      output_rank_or_simple_broadcast, fdm_H,
                                      fdm_C, lhs_padded_strides,
                                      rhs_padded_strides, fdm_output_strides);

    Impl_General<CudaT, variadic_elementwise_ops::Sum>(
        stream, output_rank_or_simple_broadcast, &lhs_padded_strides,
        reinterpret_cast<const CudaT *>(data_output), &rhs_padded_strides,
        reinterpret_cast<const CudaT *>(data_input[index]), &fdm_output_strides,
        fdm_H, fdm_C, reinterpret_cast<CudaT *>(data_output),
        output->dimension().size());
  }

  return 1;
}

#define DECLARE_SUM(Type, DataType)                                \
  template int NoBroadcastBatchImplDispatchTarget<Type, DataType>( \
      cudaStream_t stream, size_t input_count, Type * *data_input, \
      Type * data_output, vector<shared_ptr<Tensor>> & inputs,     \
      shared_ptr<Tensor> output);                                  \
  template int BinaryImplDispatchTarget<Type, DataType>(           \
      cudaStream_t stream, size_t input_count, Type * *data_input, \
      Type * data_output, vector<shared_ptr<Tensor>> & inputs,     \
      shared_ptr<Tensor> output);                                  \
  template int GeneralImplDispatchTarget<Type, DataType>(          \
      cudaStream_t stream, size_t input_count, Type * *data_input, \
      Type * data_output, vector<shared_ptr<Tensor>> & inputs,     \
      shared_ptr<Tensor> output);

DECLARE_SUM(double, CUDNN_DATA_DOUBLE)
DECLARE_SUM(float, CUDNN_DATA_FLOAT)
DECLARE_SUM(half, CUDNN_DATA_HALF)

}  // namespace Cudnn
