#ifndef CUDNN_OPS_RESIZE_HPP
#define CUDNN_OPS_RESIZE_HPP

#include <cudnn.h>

#include <memory>
#include <string>
#include <vector>

#include "cudnn/common/tensor_shape.hpp"
#include "cudnn/ops/upsamplebase.hpp"
#include "network/descriptor.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cudnn_operation.hpp"

namespace Cudnn {
template <typename Type, cudnnDataType_t DataType>
class Resize : public CudnnOperation<Type> {
 public:
  friend class OpTest;

  static std::unique_ptr<CudnnOperation<Type>> Create();
  std::shared_ptr<Tensor> Forward(cudnnHandle_t &handle, Layer &layer) override;

 private:
  void AllocateMemory();
  void OperationForward();
  void GetOutput();
  void DeAllocateMemory();
  std::vector<std::shared_ptr<Tensor>> inputs_;
  std::shared_ptr<Tensor> output_;
  cudnnHandle_t handle_;
  void *data_input_[4];
  Type *data_output_;

  size_t inputs_count_;

  UpsampleMode mode_;
  ResizeCoordinateTransformationMode coordinate_transform_mode_;
  GetOriginalCoordinateFunc get_original_coordinate_;
  ResizeNearestMode nearest_mode_;
  GetNearestPixelFunc get_nearest_pixel_;
  float cubic_coeff_a_;
  bool exclude_outside_;
  float extrapolation_value_;
  bool use_nearest2x_optimization_ = false;

  std::vector<float> scales_;
  std::vector<float> roi_;
  bool scales_cached_;
  bool roi_cached_;
  bool need_roi_input_;
  bool use_extrapolation_;
  bool is_resize_ = false;

  int roi_input_idx_ = -1;
  int scales_input_idx_ = -1;
  int sizes_input_idx_ = -1;

  UpsampleMode StringToUpsampleMode(const std::string &mode);
  ResizeCoordinateTransformationMode StringToCoordinateTransformationMode(
      const std::string &coordinate_transform_mode_name);
  GetOriginalCoordinateFunc GetOriginalCoordinateFromResizedCoordinate(
      ResizeCoordinateTransformationMode coordinate_transform_mode);
  ResizeNearestMode StringToNearestMode(const std::string &nearest_mode_name);
  GetNearestPixelFunc GetNearestPixelFromOriginal(
      ResizeNearestMode nearest_mode);
  void ScalesValidation(const std::vector<float> &scales,
                        const UpsampleMode mode) const;
  void ParseScalesData(const float *scale_data, const int64_t scale_size,
                       std::vector<float> &scales) const;
  void ParseRoiData(const float *roi_dadta, const int64_t roi_size,
                    std::vector<float> &roi_array) const;
  void ParseScalesDataFromOutputSize(gsl::span<const int64_t> output_dims,
                                     gsl::span<const int64_t> input_dims,
                                     std::vector<float> &scales) const;
  void ComputeOutputShape(const std::vector<float> &scales,
                          gsl::span<const int64_t> input_dims,
                          TensorShapeVector &output_dims) const;

  void SetOptions(Layer &layer);

  bool BaseCompute(const std::vector<float> &roi,
                   const std::vector<float> &scales,
                   const gsl::span<const int64_t> &output_dims);
};

}  // namespace Cudnn

#endif  // CUDNN_OPS_RESIZE_HPP
