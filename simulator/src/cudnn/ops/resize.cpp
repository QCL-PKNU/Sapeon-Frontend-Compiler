#include "gsl-lite.hpp"

#define NONE

#include "cudnn/common/common.cuh"
#include "cudnn/common/cuda_common.hpp"
#include "cudnn/common/cuda_utils.hpp"
#include "cudnn/common/cudnn_common.hpp"
#include "cudnn/common/ort_common.hpp"
#include "cudnn/common/utils.hpp"
#include "cudnn/ops/resize.hpp"
#include "cudnn/ops/resize_impl.hpp"
#include "cudnn/ops/upsample_impl.hpp"
#include "cudnn/ops/upsamplebase.hpp"

#define BASE CudnnOperation
#define NAME Resize
#define CLASS Cudnn::NAME
#define SCOPE CLASS<Type, DataType>
#define DB double
#define FL float
#define UC uint8_t
#define SC int8_t
#define FP64 DB, CUDNN_DATA_DOUBLE
#define FP32 FL, CUDNN_DATA_FLOAT
#define FP16 FL, CUDNN_DATA_HALF
#define UINT8 UC, CUDNN_DATA_UINT8
#define INT8 SC, CUDNN_DATA_INT8
#define STR(x) #x
#define GET_STR(x) STR(x)

#include <cassert>
#include <memory>
using std::make_shared;
using std::make_unique;
using std::shared_ptr;
using std::unique_ptr;
#include <string>
using std::string;
#include <vector>
using std::vector;
#include <cudnn.h>

#include "datatype.hpp"
#include "factory.hpp"
#include "glog/logging.h"
#include "network/descriptor.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cudnn_operation.hpp"
#include "utility.hpp"

namespace Cudnn {

static bool kRegistered = Factory<BASE<DB>>::RegisterCreateFunction(
                              GET_STR(NAME), CLASS<FP64>::Create) &&
                          Factory<BASE<FL>>::RegisterCreateFunction(
                              GET_STR(NAME), CLASS<FP32>::Create) &&
                          Factory<BASE<FL>>::RegisterCreateFunction(
                              GET_STR(NAME), CLASS<FP16>::Create) &&
                          Factory<BASE<UC>>::RegisterCreateFunction(
                              GET_STR(NAME), CLASS<UINT8>::Create);

template <typename Type, cudnnDataType_t DataType>
unique_ptr<BASE<Type>> SCOPE::Create() {
  return make_unique<CLASS<Type, DataType>>();
}

template <typename Type, cudnnDataType_t DataType>
shared_ptr<Tensor> SCOPE::Forward(cudnnHandle_t& handle, Layer& layer) {
  if (layer.intermediate_activation() == nullptr) {
    inputs_ = layer.inputs();
  } else {
    inputs_ = vector<shared_ptr<Tensor>>();
    inputs_.push_back(layer.intermediate_activation());
  }
  handle_ = handle;

  inputs_count_ = layer.predecessors().size();
  if (inputs_count_ == 0) {
    inputs_count_ = 1;
  }

  SetOptions(layer);

  OperationForward();
  GetOutput();
  DeAllocateMemory();

  return output_;
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::SetOptions(Layer& layer) {
  scales_cached_ = false;
  roi_cached_ = false;
  use_extrapolation_ = false;
  is_resize_ = true;

  mode_ = StringToUpsampleMode(layer.mode());
  if (layer.extrapolation_value() == std::numeric_limits<float>::quiet_NaN()) {
    extrapolation_value_ = 0.f;
  } else {
    extrapolation_value_ = layer.extrapolation_value();
  }

  coordinate_transform_mode_ = StringToCoordinateTransformationMode(
      layer.coordinate_transformation_mode());
  get_original_coordinate_ =
      GetOriginalCoordinateFromResizedCoordinate(coordinate_transform_mode_);
  use_extrapolation_ = need_roi_input_ =
      (coordinate_transform_mode_ == TF_CROP_AND_RESIZE);

  std::string nearest_mode_name = layer.nearest_mode();
  nearest_mode_name =
      (mode_ == NN) ? (nearest_mode_name.length() == 0 ? "round_prefer_floor"
                                                       : nearest_mode_name)
                    : "";
  nearest_mode_ = StringToNearestMode(nearest_mode_name);
  get_nearest_pixel_ = GetNearestPixelFromOriginal(nearest_mode_);

  if (layer.cubic_coeff_a() == std::numeric_limits<float>::quiet_NaN()) {
    cubic_coeff_a_ = -0.75f;
  } else {
    cubic_coeff_a_ = layer.cubic_coeff_a();
  }

  if (layer.exclude_outside() == std::numeric_limits<float>::lowest()) {
    exclude_outside_ = 0;
  } else {
    exclude_outside_ = layer.exclude_outside();
  }

  if (exclude_outside_ == 1 && mode_ != CUBIC) {
    LOG(ERROR) << "exclude_outside can be set to 1 only when mode is CUBIC. "
                  "Current mode is set to "
               << mode_;
    assert(false);
  }

  use_nearest2x_optimization_ =
      (mode_ == UpsampleMode::NN &&
       coordinate_transform_mode_ ==
           ResizeCoordinateTransformationMode::ASYMMETRIC &&
       nearest_mode_ == ResizeNearestMode::FLOOR);

  roi_input_idx_ = 1;
  scales_input_idx_ = 2;
  sizes_input_idx_ = 3;

  if (scales_input_idx_ > 0) {
    if (scales_input_idx_ < inputs_count_) {
      if (inputs_[scales_input_idx_]->dimension().size() > 0) {
        ParseScalesData((float*)inputs_[scales_input_idx_]->data(),
                        inputs_[scales_input_idx_]->dimension().size(),
                        scales_);
        scales_cached_ = true;
      }
    }
  }

  if (roi_input_idx_ > 0 && need_roi_input_) {
    if (roi_input_idx_ < inputs_count_) {
      if (inputs_[roi_input_idx_]->dimension().size() > 0) {
        ParseRoiData((float*)inputs_[roi_input_idx_]->data(),
                     inputs_[roi_input_idx_]->dimension().size(), roi_);
        roi_cached_ = true;
      }
    }
  }
}

template <typename Type, cudnnDataType_t DataType>
bool SCOPE::BaseCompute(const std::vector<float>& roi,
                        const std::vector<float>& scales,
                        const gsl::span<const int64_t>& output_dims) {
  auto X_dims = inputs_[0]->dimension().dims();
  int32_t rank = static_cast<int32_t>(X_dims.size());

  ORT_ENFORCE(static_cast<int32_t>(output_dims.size()) == rank,
              "Rank of input and output tensor should be same.");

  if (rank == 0) {
    LOG(ERROR) << "Resize: input tensor cannot be scalar.";
    return false;
  }

  if (rank != static_cast<int32_t>(scales.size())) {
    LOG(ERROR) << "Resize: input tensor's dimension does not match the scales."
               << ":" << rank;
    return false;
  }

  if (roi.size() != 2 * inputs_[0]->dimension().size()) {
    LOG(ERROR)
        << "Resize: size of roi array should be 2 * N where N is the rank of "
           "input tensor X.";
    return false;
  }

  std::vector<int64_t> calc_out_dimensions;

  calc_out_dimensions.resize(output_dims.size());
  memcpy((void*)calc_out_dimensions.data(), output_dims.data(),
         output_dims.size() * sizeof(int64_t));

  output_ =
      std::make_shared<Tensor>(calc_out_dimensions, dty::GetDataType<Type>());

  AllocateMemory();

  if (output_dims.size() == 0) {
    return true;
  }

  typedef typename ToCudaType<Type>::MappedType CudaT;

  TensorPitches input_pitches(X_dims);
  TArray<int64_t> input_strides(input_pitches);

  TensorPitches output_pitches(output_dims);
  TArray<fast_divmod> output_div_pitches(rank);

  for (int32_t i = 0; i < rank; ++i) {
    output_div_pitches[i] =
        fast_divmod(gsl::narrow_cast<int>(output_pitches[i]));
  }

  size_t output_count = output_->dimension().size();

  cudaStream_t stream;

  cudnnGetStream(handle_, &stream);

  if (is_resize_) {
    TArray<int64_t> input_shape(X_dims);
    TArray<int64_t> output_shape(output_dims);
    TArray<float, 10> roi_vals(roi);
    TArray<float> scales_vals(scales);

    size_t temp_buffer_size = CalcResizeBufferSize(mode_, output_dims);
    unsigned char* dims_mapping_buffer;

    assert(cudaMalloc(&dims_mapping_buffer,
                      temp_buffer_size * sizeof(unsigned char)) == 0);
    void* dims_mapping = reinterpret_cast<void*>(dims_mapping_buffer);

    ResizeImpl(stream, mode_, (int)rank, input_shape, output_shape,
               input_strides, output_div_pitches, scales_vals, roi_vals,
               reinterpret_cast<const CudaT*>(data_input_[0]),
               reinterpret_cast<CudaT*>(data_output_), output_count,
               use_extrapolation_,
               ToCudaType<Type>::FromFloat(extrapolation_value_),
               cubic_coeff_a_, exclude_outside_, coordinate_transform_mode_,
               nearest_mode_, dims_mapping);
    cudaFree(dims_mapping_buffer);
  } else {
    TArray<fast_divmod> scales_div(rank);

    for (int32_t i = 0; i < rank; ++i) {
      scales_div[i] = fast_divmod(gsl::narrow_cast<int>(ceil(scales[i])));
    }

    UpampleImpl(stream, mode_, rank,
                (UpsampleMode::LINEAR == mode_)
                    ? (rank == 2 ? X_dims[0] : X_dims[2])
                    : 0,
                input_strides, output_div_pitches, scales_div,
                reinterpret_cast<const CudaT*>(data_input_[0]),
                reinterpret_cast<CudaT*>(data_output_), output_count);
  }

  return true;
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::AllocateMemory() {
  for (size_t index = 0; index < inputs_count_; index++) {
    cudaMalloc(&(data_input_[index]), inputs_[index]->size());
    cudaMemcpy(data_input_[index], inputs_[index]->data(),
               inputs_[index]->size(), cudaMemcpyHostToDevice);
  }

  cudaMalloc(&data_output_, output_->size());
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::OperationForward() {
  cudaStream_t stream;

  cudnnGetStream(handle_, &stream);

  TensorShapeVector output_dims(inputs_[0]->dimension().dims().size());
  std::vector<float> roi_array(inputs_[0]->dimension().dims().size() * 2, 0.0f);
  TensorShape x_shape(inputs_[0]->dimension().dims());

  if (!roi_cached_) {
    bool use_default_roi = true;
    if (need_roi_input_) {
      ORT_ENFORCE(roi_input_idx_ > 0, "Invalid roi input index.");
      if (roi_input_idx_ <= inputs_count_) {
        ParseRoiData((float*)inputs_[roi_input_idx_]->data(),
                     inputs_[roi_input_idx_]->dimension().size(), roi_array);
        use_default_roi = false;
      }
    }
    if (use_default_roi) {
      // default roi includes ensures all the values in that axis are included
      // in the roi normalized roi is thus : [start, end] = [0, 1]
      const auto input_dims = inputs_[0]->dimension().dims();
      size_t input_rank = input_dims.size();
      roi_array.resize(input_rank * 2);
      for (size_t i = 0; i < input_rank; ++i) {
        roi_array[i] = 0;
        roi_array[i + input_rank] = 1;
      }
    }
  }

  const std::vector<float>& roi = roi_cached_ ? roi_ : roi_array;

  if (inputs_count_ == 1) {
    // Compute output shape from scales and input dims
    ComputeOutputShape(scales_, inputs_[0]->dimension().dims(), output_dims);
    assert(BaseCompute(roi, scales_, output_dims));
    return;
  }

  if (scales_cached_) {
    ORT_ENFORCE(scales_input_idx_ <= inputs_count_,
                "Only one of scales or sizes must be provided as input.");
    ComputeOutputShape(scales_, inputs_[0]->dimension().dims(), output_dims);
    assert(BaseCompute(roi, scales_, output_dims));
    return;
  }

  std::vector<float> scales_array(inputs_[0]->dimension().dims().size());
  if (scales_input_idx_ <= inputs_count_ &&
      inputs_[scales_input_idx_]->dimension().dims().size() != 0) {
    // use scales input data
    ORT_ENFORCE(sizes_input_idx_ <= inputs_count_,
                "Only one of scales or sizes must be provided as input.");

    LOG(ERROR) << inputs_[scales_input_idx_]->dimension().size();
    ParseScalesData((float*)inputs_[scales_input_idx_]->data(),
                    inputs_[scales_input_idx_]->dimension().size(),
                    scales_array);
    ComputeOutputShape(scales_array, x_shape.GetDims(), output_dims);
  } else {
    // When sizes input is available directly populate it into the output_dims
    // array.
    ORT_ENFORCE(sizes_input_idx_ <= inputs_count_ &&
                    inputs_[sizes_input_idx_]->dimension().dims().size() != 0,
                "Either scales or sizes MUST be provided as input.");
    ORT_ENFORCE(
        inputs_[sizes_input_idx_]->dimension().dims().size() ==
            static_cast<int64_t>(output_dims.size()),
        "Resize: input tensor's rank does not match the output tensor's rank.");
    memcpy(
        output_dims.data(), (int64_t*)inputs_[sizes_input_idx_]->data(),
        inputs_[sizes_input_idx_]->dimension().dims().size() * sizeof(int64_t));

    ParseScalesDataFromOutputSize(output_dims, x_shape.GetDims(), scales_array);
  }

  assert(BaseCompute(roi, scales_array, output_dims));
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::GetOutput() {
  cudaMemcpy(output_->data(), data_output_, output_->size(),
             cudaMemcpyDeviceToHost);
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::DeAllocateMemory() {
  cudaFree(data_output_);
  cudaFree(data_input_[0]);
  cudaFree(data_input_[1]);
}

template <typename Type, cudnnDataType_t DataType>
UpsampleMode SCOPE::StringToUpsampleMode(const std::string& mode) {
  if (mode == UpsampleModeNN) {
    return UpsampleMode::NN;
  }
  if (mode == UpsampleModeLinear) {
    return UpsampleMode::LINEAR;
  }
  if (mode == UpsampleModeCubic) {
    return UpsampleMode::CUBIC;
  }
  LOG(ERROR) << "mode attribute is " << mode << ". It can only be "
             << UpsampleModeNN << "(default) or " << UpsampleModeLinear
             << " or " << UpsampleModeCubic << ".";
  assert(false);
}

template <typename Type, cudnnDataType_t DataType>
ResizeCoordinateTransformationMode SCOPE::StringToCoordinateTransformationMode(
    const std::string& coordinate_transform_mode_name) {
  if (coordinate_transform_mode_name == "asymmetric") {
    return ASYMMETRIC;
  }
  if (coordinate_transform_mode_name == "pytorch_half_pixel") {
    return PYTORCH_HALF_PIXEL;
  }
  if (coordinate_transform_mode_name == "tf_half_pixel_for_nn") {
    return TF_HALF_PIXEL_FOR_NN;
  }
  if (coordinate_transform_mode_name == "align_corners") {
    return ALIGN_CORNERS;
  }
  if (coordinate_transform_mode_name == "tf_crop_and_resize") {
    return TF_CROP_AND_RESIZE;
  }
  if (coordinate_transform_mode_name == "half_pixel") {
    return HALF_PIXEL;
  }
  LOG(ERROR) << "coordinate_transform_mode:[" << coordinate_transform_mode_name
             << "] is not supportted!";
  assert(false);
}

template <typename Type, cudnnDataType_t DataType>
GetOriginalCoordinateFunc SCOPE::GetOriginalCoordinateFromResizedCoordinate(
    ResizeCoordinateTransformationMode coordinate_transform_mode) {
  switch (coordinate_transform_mode) {
    case ASYMMETRIC:
      return [](float x_resized, float x_scale, float, float, float, float) {
        return x_resized / x_scale;
      };
    case PYTORCH_HALF_PIXEL:
      return [](float x_resized, float x_scale, float length_resized, float,
                float, float) {
        return length_resized > 1 ? (x_resized + 0.5f) / x_scale - 0.5f : 0.0f;
      };
    case TF_HALF_PIXEL_FOR_NN:
      return [](float x_resized, float x_scale, float, float, float, float) {
        return (x_resized + 0.5f) / x_scale;
      };
    case ALIGN_CORNERS:
      return [](float x_resized, float, float length_resized,
                float length_original, float, float) {
        return length_resized == 1
                   ? 0
                   : x_resized * (length_original - 1) / (length_resized - 1);
      };
    case TF_CROP_AND_RESIZE:
      return [](float x_resized, float, float length_resized,
                float length_original, float roi_start, float roi_end) {
        auto orig = length_resized > 1
                        ? roi_start * (length_original - 1) +
                              (x_resized * (roi_end - roi_start) *
                               (length_original - 1)) /
                                  (length_resized - 1)
                        : 0.5 * (roi_start + roi_end) * (length_original - 1);
        return static_cast<float>(orig);
      };
    default:  // "half_pixel"
      return [](float x_resized, float x_scale, float, float, float, float) {
        return ((x_resized + 0.5f) / x_scale) - 0.5f;
      };
  }
}

template <typename Type, cudnnDataType_t DataType>
ResizeNearestMode SCOPE::StringToNearestMode(
    const std::string& nearest_mode_name) {
  if (nearest_mode_name == "round_prefer_floor") {
    return ROUND_PREFER_FLOOR;
  } else if (nearest_mode_name == "round_prefer_ceil") {
    return ROUND_PREFER_CEIL;
  } else if (nearest_mode_name == "floor") {
    return FLOOR;
  } else if (nearest_mode_name == "ceil") {
    return CEIL;
  } else if (nearest_mode_name == "") {
    return SIMPLE;
  }
  LOG(ERROR) << "nearest_mode:[" << nearest_mode_name << "] is not supported!";
  assert(false);
}

template <typename Type, cudnnDataType_t DataType>
GetNearestPixelFunc SCOPE::GetNearestPixelFromOriginal(
    ResizeNearestMode nearest_mode) {
  switch (nearest_mode) {
    case SIMPLE:
      // versions older than 11 did not have nearest_mode attr. Use the
      // original logic in this case to maintain backward compatibility
      return [](float x_original, bool isDownSample) {
        if (isDownSample) {
          return static_cast<int64_t>(std::ceil(x_original));
        } else {
          return static_cast<int64_t>(x_original);
        }
      };
    case ROUND_PREFER_CEIL:
      return [](float x_original, bool) {
        return static_cast<int64_t>(std::round(x_original));
      };
    case FLOOR:
      return [](float x_original, bool) {
        return static_cast<int64_t>(std::floor(x_original));
      };
    case CEIL:
      return [](float x_original, bool) {
        return static_cast<int64_t>(std::ceil(x_original));
      };
    default:  // default is round_prefer_floor
      return [](float x_original, bool) {
        // for half way cases prefer floor
        if (x_original == static_cast<int64_t>(x_original) + 0.5f) {
          return static_cast<int64_t>(std::floor(x_original));
        }
        return static_cast<int64_t>(std::round(x_original));
      };
  }
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::ScalesValidation(const std::vector<float>& scales,
                             const UpsampleMode mode) const {
  if (!is_resize_) {
    for (auto& scale : scales) {
      ORT_ENFORCE(scale >= 1,
                  "Scale value should be greater than or equal to 1.");
    }
  } else {
    for (auto& scale : scales) {
      ORT_ENFORCE(scale > 0, "Scale value should be greater than 0.");
    }
  }

  if (UpsampleMode::LINEAR == mode) {
    ORT_ENFORCE(
        scales.size() == 2 ||
            (scales.size() == 4 && scales[0] == 1 && scales[1] == 1) ||
            (scales.size() == 4 && scales[0] == 1 && scales[3] == 1) ||
            scales.size() == 3 ||
            (scales.size() == 5 && scales[0] == 1 && scales[1] == 1),
        "'Linear' mode only support:\n"
        "  * 2-D inputs or\n"
        "  * 3-D inputs ('Bilinear', 'Trilinear') or\n"
        "  * 4-D inputs with the corresponding outermost 2 scale values "
        "being 1"
        " or the corresponding outermost and innermost scale values being 1 "
        "or\n"
        "  * 5-D inputs with the corresponding outermost 2 scale values "
        "being 1"
        "in the Resize operator")
  } else if (UpsampleMode::CUBIC == mode) {
    ORT_ENFORCE(
        scales.size() == 2 ||
            (scales.size() == 4 && scales[0] == 1 && scales[1] == 1),
        "'Cubic' mode only support 2-D inputs ('Bicubic') or 4-D inputs "
        "with the corresponding outermost 2 scale values being 1 in the "
        "Resize operator");
  }
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::ParseScalesData(const float* scale_data, const int64_t scales_size,
                            std::vector<float>& scales) const {
  ORT_ENFORCE(scales_size > 0, "scales size should be greater than 0.");
  if (scales.empty()) {
    scales.resize(scales_size);
  }
  memcpy(scales.data(), scale_data, scales_size * sizeof(float));
  ScalesValidation(scales, mode_);
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::ParseRoiData(const float* roi_dadta, const int64_t roi_size,
                         std::vector<float>& roi_array) const {
  if (roi_size > 0) {
    roi_array.resize(roi_size);
    memcpy(roi_array.data(), roi_dadta, roi_size * sizeof(float));
  }
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::ParseScalesDataFromOutputSize(gsl::span<const int64_t> output_dims,
                                          gsl::span<const int64_t> input_dims,
                                          std::vector<float>& scales) const {
  for (size_t i = 0, end = input_dims.size(); i < end; ++i) {
    // Handle corner case to avoid dividing by zero in the next step
    if (input_dims[i] == 0) {
      // Enforce that output_dim is 0, given that we cannot scale 0 by any
      // factor to result in any non-zero value
      ORT_ENFORCE(
          output_dims[i] == 0,
          "Input dim is zero but required output dim is non-zero. "
          "Cannot scale 0 by any factor to generate a non-zero value. ");
      /*
              "Dimension: " +
                  std::string(i) +
                  " Input dim value: " + std::string(input_dims[i]) +
                  " Output dim value: " + std::string(output_dims[i]));
                  */
      // Scale can be any arbitrary value as technically scaling 0 by any
      // factor results in 0. Keeping scale as 1 is more intuitive given that
      // input_dim
      // == output_dim.
      scales[i] = 1.f;
    } else {
      scales[i] = static_cast<float>(output_dims[i]) /
                  static_cast<float>(input_dims[i]);
    }
  }
  ScalesValidation(scales, mode_);
}

template <typename Type, cudnnDataType_t DataType>
void SCOPE::ComputeOutputShape(const std::vector<float>& scales,
                               gsl::span<const int64_t> input_dims,
                               TensorShapeVector& output_dims) const {
  for (std::size_t i = 0; i < input_dims.size(); i++) {
    output_dims[i] = static_cast<int64_t>(scales[i] * input_dims[i]);
  }
}

}  // namespace Cudnn
