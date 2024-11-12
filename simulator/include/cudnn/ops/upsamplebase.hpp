// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CUDNN_OPS_UPSAMPLEBASE_HPP
#define CUDNN_OPS_UPSAMPLEBASE_HPP

namespace Cudnn {

constexpr const char* UpsampleModeNN = "nearest";
constexpr const char* UpsampleModeLinear = "linear";
constexpr const char* UpsampleModeCubic = "cubic";

using GetNearestPixelFunc = int64_t (*)(float, bool);
using GetOriginalCoordinateFunc = float (*)(float, float, float, float, float,
                                            float);

enum UpsampleMode {
  NN = 0,      // nearest neighbor
  LINEAR = 1,  // linear interpolation
  CUBIC = 2,   // cubic interpolation
};

enum ResizeCoordinateTransformationMode {
  HALF_PIXEL = 0,
  ASYMMETRIC = 1,
  PYTORCH_HALF_PIXEL = 2,
  TF_HALF_PIXEL_FOR_NN = 3,
  ALIGN_CORNERS = 4,
  TF_CROP_AND_RESIZE = 5,
  CoordinateTransformationModeCount = 6,
};

enum ResizeNearestMode {
  SIMPLE = 0,  // For resize op 10
  ROUND_PREFER_FLOOR = 1,
  ROUND_PREFER_CEIL = 2,
  FLOOR = 3,
  CEIL = 4,
  NearestModeCount = 5,
};

}  // namespace Cudnn

#endif  // CUDNN_OPS_UPSAMPLEBASE_HPP
