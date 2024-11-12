// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cudnn/common/reduction_ops.hpp"

#include <glog/logging.h>

#include "cudnn/common/binary_elementwise.hpp"
#include "cudnn/common/binary_elementwise_ops.hpp"
#include "cudnn/common/cuda_common.hpp"
#include "cudnn/common/cudnn_common.hpp"
#include "cudnn/common/reduction_functions.hpp"
#include "cudnn/common/tensor_shape.hpp"
#include "cudnn/common/unary_elementwise_ops_impl.cuh"

namespace Cudnn {

// `input_shape_override` (if provided) is the input shape for compute purposes
bool PrepareForReduce(TensorShape x_shape, bool keepdims,
                      gsl::span<const int64_t> axes,
                      PrepareReduceMetadata& prepare_reduce_metadata,
                      const TensorShape* input_shape_override) {
  assert(x_shape.Size() != 0);

  const TensorShape& input_shape =
      input_shape_override ? *input_shape_override : x_shape;
  const int64_t rank = gsl::narrow<int64_t>(input_shape.NumDimensions());
  prepare_reduce_metadata.input_count = input_shape.Size();

  if (rank > 8) {
    LOG(ERROR) << "cuDNN only supports up to 8-D tensors in reduction";
    return false;
  }

  const auto input_dims = input_shape.GetDims();
  std::vector<bool> reduced(rank, false);
  if (axes.size() > 0) {
    prepare_reduce_metadata.output_dims = input_shape.AsShapeVector();
    for (auto axis : axes) {
      axis = HandleNegativeAxis(axis, rank);
      if (!(input_dims[axis] != 0)) {
        LOG(ERROR)
            << "Can't reduce on dim with value of 0 if 'keepdims' is false. "
            << "Invalid output shape would be produced";
      }
      assert(input_dims[axis] != 0);
      prepare_reduce_metadata.output_dims[axis] = 1;
      reduced[axis] = true;
    }
  } else {
    // no axes provided (i.e.) default axes  => reduce on all dims
    prepare_reduce_metadata.output_dims.reserve(input_dims.size());
    for (auto dim : input_dims) {
      if (!(keepdims || dim != 0)) {
        LOG(ERROR)
            << "Can't reduce on dim with value of 0 if 'keepdims' is false. "
            << "Invalid output shape would be produced" << input_shape;
      }
      assert(keepdims || dim != 0);
      prepare_reduce_metadata.output_dims.push_back(dim == 0 ? 0 : 1);
    }
  }

  if (keepdims) {
    prepare_reduce_metadata.squeezed_output_dims =
        prepare_reduce_metadata.output_dims;
  } else if (axes.size() > 0) {
    // we are not going to keep the reduced dims, hence compute the final output
    // dim accordingly
    prepare_reduce_metadata.squeezed_output_dims.reserve(
        rank);  // even though we won't use the full capacity, it is better to
                // reserve for peak possible usage
    for (auto i = 0; i < rank; ++i) {
      if (!reduced[i])
        prepare_reduce_metadata.squeezed_output_dims.push_back(input_dims[i]);
    }
  } else {
    // 'axes' is empty and keepdims is false => we reduce on all axes AND drop
    // all dims, so the result is just a scalar, we keep 'squeezed_output_dims'
    // empty (i.e.) no-op
  }

  // CUDNN requires at least 3D input, so pad 1s if needed
  prepare_reduce_metadata.input_dims_cudnn = input_shape.AsShapeVector();
  prepare_reduce_metadata.output_dims_cudnn =
      prepare_reduce_metadata.output_dims;
  if (rank < 3) {
    TensorShapeVector pads(3 - rank, 1);
    prepare_reduce_metadata.input_dims_cudnn.insert(
        prepare_reduce_metadata.input_dims_cudnn.end(), pads.begin(),
        pads.end());
    prepare_reduce_metadata.output_dims_cudnn.insert(
        prepare_reduce_metadata.output_dims_cudnn.end(), pads.begin(),
        pads.end());
  }

  prepare_reduce_metadata.output_count =
      TensorShape(prepare_reduce_metadata.output_dims).Size();

  return true;
}

// `input_shape_override` is the input shape for compute purposes (if provided)
template <typename T, cudnnReduceTensorIndices_t ReduceTensorIndices>
bool ReduceComputeCore(cudaStream_t stream, cudnnHandle_t cudnn_handle,
                       const TensorShape& x_shape, T* x_data,
                       PrepareReduceMetadata& prepare_reduce_metadata,
                       TensorShape& output_shape, T* output_data,
                       cudnnReduceTensorOp_t cudnn_reduce_op,
                       gsl::span<const int64_t> axes, bool calculate_log,
                       bool calculate_sqt, bool log_sum_exp,
                       bool fast_reduction, int out_type,
                       const TensorShape* input_shape_override) {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const TensorShape& input_shape =
      input_shape_override ? *input_shape_override : x_shape;

  int64_t input_count = prepare_reduce_metadata.input_count;
  int64_t output_count = prepare_reduce_metadata.output_count;
  auto& output_dims = prepare_reduce_metadata.output_dims;
  auto& input_dims_cudnn = prepare_reduce_metadata.input_dims_cudnn;
  auto& output_dims_cudnn = prepare_reduce_metadata.output_dims_cudnn;

  // special case when there is a dim value of 0 in the shape.
  if (input_count == 0) {
    assert(output_shape.Size() == 0);
    return true;
  }

  // Block of fast matrix reduction.
  if (fast_reduction) {
    int m{}, n{};
    T* input_data_buffer = nullptr;
    const auto applicable_matrix_reduction = get_applicable_matrix_reduction(
        cudnn_reduce_op, input_shape.GetDims(), axes, m, n);
    if (applicable_matrix_reduction != ApplicableMatrixReduction::None) {
      const CudaT* input_data = reinterpret_cast<const CudaT*>(x_data);
      if (calculate_sqt) {
        cudaMalloc(&input_data_buffer, input_count * sizeof(T));
        input_data = reinterpret_cast<CudaT*>(input_data_buffer);
        fast_divmod tmp_div;
        Impl_Mul<CudaT>(
            stream, static_cast<int32_t>(SimpleBroadcast::NoBroadcast), nullptr,
            reinterpret_cast<const CudaT*>(x_data), nullptr,
            reinterpret_cast<const CudaT*>(x_data), nullptr, tmp_div, tmp_div,
            reinterpret_cast<CudaT*>(input_data_buffer), input_count);
        input_data = reinterpret_cast<const CudaT*>(input_data_buffer);
      }

      switch (applicable_matrix_reduction) {
        case ApplicableMatrixReduction::Rows: {
          assert(reduce_matrix_rows(
              stream, input_data, reinterpret_cast<CudaT*>(output_data), m, n));
        } break;
        case ApplicableMatrixReduction::Columns: {
          const auto buffer_size_bytes =
              compute_reduce_matrix_columns_buffer_size<CudaT>(m, n);
          CudaT* buffer;
          cudaMalloc(&buffer, buffer_size_bytes);
          assert(reduce_matrix_columns(stream, input_data,
                                       reinterpret_cast<CudaT*>(output_data), m,
                                       n, buffer, buffer_size_bytes));
          cudaFree(buffer);
        } break;
        default: {
          LOG(ERROR) << "Invild matrix reduction type.";
          assert(false);
        }
      }

      if (calculate_log) {
        Impl_Log<CudaT>(stream, reinterpret_cast<const CudaT*>(output_data),
                        reinterpret_cast<CudaT*>(output_data), output_count);
      } else if (cudnn_reduce_op == CUDNN_REDUCE_TENSOR_AVG) {
        float denominator_float =
            applicable_matrix_reduction == ApplicableMatrixReduction::Rows
                ? static_cast<float>(m)
                : static_cast<float>(n);
        CudaT denominator = ToCudaType<T>::FromFloat(denominator_float);
        UnaryDiv(stream, reinterpret_cast<const CudaT*>(output_data),
                 reinterpret_cast<CudaT*>(output_data), denominator,
                 output_count);
      }

      if (input_data_buffer != nullptr) {
        cudaFree(input_data_buffer);
      }

      return true;
    }
  }

  // This reduction keep adding values to this buffer. If a non-zero value, say
  // 1000, is here, the sum will start with 1000. Therefore zeroing out the
  // memory is required
  assert(cudaMemsetAsync(output_data, 0,
                         output_shape.Size() * sizeof(T) * out_type,
                         stream) == 0);

  float* temp_X = nullptr;
  cudnnDataType_t cudnn_type_X = CUDNN_DATA_FLOAT;

  cudnn_type_X = CudnnTensor::GetDataType<CudaT>();

  CudnnReduceDescriptor reduce_desc;
  assert(reduce_desc.Set(cudnn_reduce_op, cudnn_type_X, ReduceTensorIndices));

  const auto one = Consts<CudaT>::One;
  const auto zero = Consts<CudaT>::Zero;
  CudnnTensor input_tensor;
  CudnnTensor output_tensor;
  assert(input_tensor.Set(input_dims_cudnn, cudnn_type_X));
  assert(output_tensor.Set(output_dims_cudnn, cudnn_type_X));
  size_t workspace_bytes = 0;
  assert(cudnnGetReductionWorkspaceSize(cudnn_handle, reduce_desc, input_tensor,
                                        output_tensor, &workspace_bytes) == 0);
  CudaT* workspace_cuda;
  cudaMalloc(&workspace_cuda, workspace_bytes * sizeof(CudaT));

  size_t indices_bytes = 0;
  assert(cudnnGetReductionIndicesSize(cudnn_handle, reduce_desc, input_tensor,
                                      output_tensor, &indices_bytes) == 0);

  uint32_t* indices_cuda;
  cudaMalloc(&indices_cuda, indices_bytes * sizeof(uint32_t));

  if (ReduceTensorIndices == CUDNN_REDUCE_TENSOR_NO_INDICES) {
    T* input_data_buffer = nullptr;
    CudaT* input_data = nullptr;
    if (calculate_sqt) {
      cudaMalloc(&input_data_buffer, input_count * sizeof(T));
      input_data = reinterpret_cast<CudaT*>(input_data_buffer);
      fast_divmod tmp_div;
      Impl_Mul<CudaT>(stream,
                      static_cast<int32_t>(SimpleBroadcast::NoBroadcast),
                      nullptr, reinterpret_cast<const CudaT*>(x_data), nullptr,
                      reinterpret_cast<const CudaT*>(x_data), nullptr, tmp_div,
                      tmp_div, input_data, input_count);
    } else if (log_sum_exp) {
      // cudnnReduceTensor for ReduceSum has issue if input and output has same
      // size, we just need to copy the data for this case This happens when the
      // input is Scalar
      if (input_count == output_count) {
        if (output_data != x_data) {
          assert(cudaMemcpyAsync(output_data, x_data, input_count * sizeof(T),
                                 cudaMemcpyDeviceToDevice, stream) == 0);
        }
      } else {
        // Reduce max -- Max/Min will output indices data
        CudnnReduceDescriptor reduce_max_desc;
        cudnnDataType_t cudnn_reduce_max_type = cudnn_type_X;
        assert(reduce_max_desc.Set(CUDNN_REDUCE_TENSOR_MAX,
                                   cudnn_reduce_max_type,
                                   CUDNN_REDUCE_TENSOR_NO_INDICES));
        size_t indices_bytes_max = 0;
        assert(cudnnGetReductionIndicesSize(cudnn_handle, reduce_max_desc,
                                            input_tensor, output_tensor,
                                            &indices_bytes_max) == 0);

        uint32_t* indices_cuda_max;
        cudaMalloc(&indices_cuda_max, indices_bytes * sizeof(uint32_t));
        assert(cudnnReduceTensor(
                   cudnn_handle, reduce_max_desc, indices_cuda_max,
                   indices_bytes_max, workspace_cuda, workspace_bytes, &one,
                   input_tensor, reinterpret_cast<const CudaT*>(x_data), &zero,
                   output_tensor, reinterpret_cast<CudaT*>(output_data)) == 0);
        cudaFree(indices_cuda_max);
      }

      // Exp(X-ReduceMax)
      const TensorShape output_shape(output_dims);

      T* exp_result_buffer = nullptr;
      cudaMalloc(&exp_result_buffer, input_count * sizeof(T));

      auto exp_result = exp_result_buffer;

      T* log_sum_result_buffer = nullptr;
      cudaMalloc(&log_sum_result_buffer, output_count * sizeof(T));

      auto log_sum_result = log_sum_result_buffer;
      BinaryElementwisePreparation prepare;
      assert(prepare.BinaryElementwiseBroadcastPrepareHelper(
          input_shape, output_shape, input_shape));
      Impl_Sub<CudaT>(
          stream, prepare.output_rank_or_simple_broadcast,
          &prepare.lhs_padded_strides, reinterpret_cast<const CudaT*>(x_data),
          &prepare.rhs_padded_strides, reinterpret_cast<CudaT*>(output_data),
          &prepare.fdm_output_strides, prepare.fdm_H, prepare.fdm_C,
          reinterpret_cast<CudaT*>(exp_result), input_count);

      Impl_Exp<CudaT>(stream, reinterpret_cast<CudaT*>(exp_result),
                      reinterpret_cast<CudaT*>(exp_result), input_count);

      // cudnnReduceTensor for ReduceSum has issue if input and output has same
      // size, we just need to copy the data for this case This happens when the
      // input is Scalar. We do not need to add anything in this case.
      if (input_count == output_count) {
        assert(cudaMemcpyAsync(reinterpret_cast<CudaT*>(log_sum_result),
                               exp_result, input_count * sizeof(T),
                               cudaMemcpyDeviceToDevice, stream) == 0);
      } else {
        // ReduceSum
        assert(cudnnReduceTensor(
                   cudnn_handle, reduce_desc, indices_cuda, indices_bytes,
                   workspace_cuda, workspace_bytes, &one, input_tensor,
                   exp_result, &zero, output_tensor,
                   reinterpret_cast<CudaT*>(log_sum_result)) == 0);
      }

      // Log(Sum)
      Impl_Log<CudaT>(stream, reinterpret_cast<CudaT*>(log_sum_result),
                      reinterpret_cast<CudaT*>(log_sum_result), output_count);

      // Log + ReduceMax
      fast_divmod tmp_div;
      Impl_Add<CudaT>(
          stream, static_cast<int32_t>(SimpleBroadcast::NoBroadcast), nullptr,
          reinterpret_cast<CudaT*>(log_sum_result), nullptr,
          reinterpret_cast<CudaT*>(output_data), nullptr, tmp_div, tmp_div,
          reinterpret_cast<CudaT*>(output_data), output_count);

      if (exp_result_buffer == nullptr) {
        cudaFree(exp_result_buffer);
      }
      if (log_sum_result_buffer == nullptr) {
        cudaFree(log_sum_result_buffer);
      }
      if (input_data_buffer == nullptr) {
        cudaFree(input_data_buffer);
      }

      cudaFree(workspace_cuda);
      cudaFree(indices_cuda);
      return true;
    }
    if (calculate_sqt) {
      // cudnnReduceTensor for ReduceSum has issue if input and output has same
      // size, we just need to copy the data for this case This happens when the
      // input is Scalar. We do not need to add anything in this case.
      if (input_count == output_count) {
        assert(cudaMemcpyAsync(reinterpret_cast<CudaT*>(output_data),
                               input_data, input_count * sizeof(T),
                               cudaMemcpyDeviceToDevice, stream) == 0);
      } else {
        assert(cudnnReduceTensor(cudnn_handle, reduce_desc, indices_cuda,
                                 indices_bytes, workspace_cuda, workspace_bytes,
                                 &one, input_tensor, input_data, &zero,
                                 output_tensor,
                                 reinterpret_cast<CudaT*>(output_data)) == 0);
      }
    } else {
      // cudnnReduceTensor for ReduceSum has issue if input and output has same
      // size, we just need to copy the data for this case
      if (input_count == output_count) {
        if (output_data != x_data) {
          assert(cudaMemcpyAsync(output_data, x_data, input_count * sizeof(T),
                                 cudaMemcpyDeviceToDevice, stream) == 0);
        }
      } else {
        if (temp_X) {
          float* temp_output;
          cudaMalloc(&temp_output, output_count * sizeof(float));
          assert(cudnnReduceTensor(cudnn_handle, reduce_desc, indices_cuda,
                                   indices_bytes, workspace_cuda,
                                   workspace_bytes, &one, input_tensor, temp_X,
                                   &zero, output_tensor, temp_output) == 0);

          Impl_Cast<float, CudaT>(stream, temp_output,
                                  reinterpret_cast<CudaT*>(output_data),
                                  output_count);
          cudaFree(temp_output);
        } else {
          assert(cudnnReduceTensor(cudnn_handle, reduce_desc, indices_cuda,
                                   indices_bytes, workspace_cuda,
                                   workspace_bytes, &one, input_tensor,
                                   reinterpret_cast<const CudaT*>(x_data),
                                   &zero, output_tensor,
                                   reinterpret_cast<CudaT*>(output_data)) == 0);
        }
      }
    }

    if (input_data_buffer == nullptr) {
      cudaFree(input_data_buffer);
    }
  } else {
    // For ArgMax & ArgMin ops, use the indicies as the output with int64 type
    // cudnnReduceTensor has issue if input and output has same size, which will
    // happen if the axis to be reduced has dim value of 1. the output is zeros
    // of the output size
    if (input_count == output_count) {
      assert(cudaMemsetAsync(output_data, static_cast<int64_t>(0),
                             output_count * sizeof(int64_t), stream) == 0);
    } else {
      if (temp_X) {
        float* temp_output;
        cudaMalloc(&temp_output, output_count * sizeof(float));
        assert(cudnnReduceTensor(cudnn_handle, reduce_desc, indices_cuda,
                                 indices_bytes, workspace_cuda, workspace_bytes,
                                 &one, input_tensor, temp_X, &zero,
                                 output_tensor, temp_output) == 0);
        cudaFree(temp_output);
      } else {
        CudaT* temp_output;
        cudaMalloc(&temp_output, output_count * sizeof(CudaT));
        assert(cudnnReduceTensor(cudnn_handle, reduce_desc, indices_cuda,
                                 indices_bytes, workspace_cuda, workspace_bytes,
                                 &one, input_tensor,
                                 reinterpret_cast<const CudaT*>(x_data), &zero,
                                 output_tensor, temp_output) == 0);
        cudaFree(temp_output);
      }

      // CUDA reduction index is uint32_t for now, cast it to int64_t according
      // to ONNX spec
      Impl_Cast<uint32_t, int64_t>(stream,
                                   reinterpret_cast<uint32_t*>(indices_cuda),
                                   (int64_t*)output_data, output_count);
    }
  }

  if (calculate_log) {
    Impl_Log<CudaT>(stream, reinterpret_cast<CudaT*>(output_data),
                    reinterpret_cast<CudaT*>(output_data), output_count);
  }

  cudaFree(workspace_cuda);
  cudaFree(indices_cuda);

  return true;
}

#define DEFINE_REDUCE_OP_TYPE1(T, method)                                   \
  template bool ReduceComputeCore<T, method>(                               \
      cudaStream_t stream, cudnnHandle_t cudnn_handle,                      \
      const TensorShape& x_shape, T* x_data,                                \
      PrepareReduceMetadata& prepare_reduce_metadata,                       \
      TensorShape& output_shape, T* output_data,                            \
      cudnnReduceTensorOp_t cudnn_reduce_op, gsl::span<const int64_t> axes, \
      bool calculate_log, bool calculate_sqt, bool log_sum_exp,             \
      bool fast_reduction, int out_type,                                    \
      const TensorShape* input_shape_override = nullptr);

DEFINE_REDUCE_OP_TYPE1(float, CUDNN_REDUCE_TENSOR_NO_INDICES)
DEFINE_REDUCE_OP_TYPE1(double, CUDNN_REDUCE_TENSOR_NO_INDICES)
DEFINE_REDUCE_OP_TYPE1(half, CUDNN_REDUCE_TENSOR_NO_INDICES)

DEFINE_REDUCE_OP_TYPE1(float, CUDNN_REDUCE_TENSOR_FLATTENED_INDICES)
DEFINE_REDUCE_OP_TYPE1(double, CUDNN_REDUCE_TENSOR_FLATTENED_INDICES)
DEFINE_REDUCE_OP_TYPE1(half, CUDNN_REDUCE_TENSOR_FLATTENED_INDICES)

}  // namespace Cudnn
