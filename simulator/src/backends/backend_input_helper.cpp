#include "backends/backend_input_helper.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "arguments.hpp"
#include "enums/error.hpp"
#include "glog/logging.h"
#include "image.hpp"
#include "network/dimension.hpp"
#include "network/tensor.hpp"
#include "tl/expected.hpp"
#include "utility.hpp"

BackendInputHelper::BackendInputHelper(Arguments& args,
                                       Dimension input_dimension)
    : dimension_(input_dimension) {
  preprocess_config_file_path_ = args.preprocess_config_path();
}

Tensor BackendInputHelper::GetInputImageTensor(
    const std::string& input_file_path) {
  assert(dimension_.dims().size() == 4);
  const auto c = dimension_.c();
  const auto h = dimension_.h();
  const auto w = dimension_.w();

  Tensor input{1, c, h, w, dty::DataType::FP32};
  float* input_data = input.data<float>();

  auto image = std::make_unique<image::Image>();
  image->LoadImageColor(input_file_path);
  image->ParsePreprocessParameter(preprocess_config_file_path_);

  auto resize_size = image->resize_size();
  if (resize_size.has_value()) {
    image->ResizeImage(resize_size.value(), resize_size.value());
  }

  auto crop_size = image->crop_size();
  if (crop_size.has_value()) {
    image->CenterCropImage(crop_size.value(), crop_size.value());
  } else {
    image->SquareImage();
    image->ResizeImage(w, h);
  }

  image->ConvertFormat();
  image->CopyData(input_data, w * h * c);

  return input;
}

Tensor BackendInputHelper::GetInputImageTensor(
    const std::string& input_file_path, float threshold) {
  assert(dimension_.dims().size() == 4);
  constexpr float kINT8Min = -127.0F;
  constexpr float kINT8Max = 127.0F;
  auto fp_input = GetInputImageTensor(input_file_path);

  assert(fp_input.dtype() == dty::DataType::FP32);
  float* fp_data = fp_input.data<float>();
  double input_scale = static_cast<double>(kINT8Max / threshold);

  Tensor input{fp_input.dimension().dims(), dty::DataType::SINT8};
  int8_t* input_data = input.data<int8_t>();

  assert(fp_input.dimension().dims().size() == input.dimension().dims().size());

  for (int i = 0; i < fp_input.dimension().size(); ++i) {
    float q = std::round(fp_data[i] * input_scale);
    q = std::min(kINT8Max, std::max(kINT8Min, q));
    input_data[i] = static_cast<int8_t>(q);
  }

  return input;
}

tl::expected<Tensor, SimulatorError> BackendInputHelper::FuseInputTensors(
    const std::vector<Tensor>& tensors) {
  const auto& first = tensors.at(0);

  size_t n = 0;
  const size_t c = first.c();
  const size_t h = first.h();
  const size_t w = first.w();
  const auto dtype = first.dtype();

  for (const auto& tensor : tensors) {
    n += tensor.n();
    if (c != tensor.c() || h != tensor.h() || w != tensor.w() ||
        dtype != tensor.dtype()) {
      LOG(ERROR) << "tensor fusion failed.\nc : " << c << ", " << tensor.c()
                 << "\nh : " << h << ", " << tensor.h() << "\nw : " << w << ", "
                 << tensor.w() << "\ndtype : " << dty::NameOf(dtype) << ", "
                 << dty::NameOf(tensor.dtype());
      return tl::make_unexpected(SimulatorError::kTensorShapeError);
    }
  }

  Tensor fused{n, c, h, w, dtype};
  void* fused_data = fused.data();
  size_t offset = 0;

  for (const auto& tensor : tensors) {
    if (dtype == dty::DataType::FP32) {
      std::memcpy(static_cast<float*>(fused_data) + offset, tensor.data(),
                  tensor.size());
    } else if (dtype == dty::DataType::SINT8) {
      std::memcpy(static_cast<int8_t*>(fused_data) + offset, tensor.data(),
                  tensor.size());
    } else {
      LOG(ERROR) << "not supported data type for input tensor : "
                 << dty::NameOf(dtype);
      return tl::make_unexpected(SimulatorError::kInvalidDataType);
    }
    offset += tensor.dimension().size();
  }

  return fused;
}
