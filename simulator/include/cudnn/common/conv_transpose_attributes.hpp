/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/* Modifications Copyright (c) Microsoft. */

#ifndef CUDNN_COMMON_CONV_TRANSPOSE_ATTRIBUTES_HPP
#define CUDNN_COMMON_CONV_TRANSPOSE_ATTRIBUTES_HPP

#include <glog/logging.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "cudnn/common/conv_attributes.hpp"
#include "cudnn/common/tensor_shape.hpp"

namespace Cudnn {

struct ConvTransposeAttributes : public ConvAttributes {
  explicit ConvTransposeAttributes() {}

  void SetConvTransposeAttributes(const std::string& auto_pad_str,
                                  const TensorShapeVector& kernel_shape,
                                  const TensorShapeVector& in_strides,
                                  const gsl::span<const int64_t> pads_span,
                                  const TensorShapeVector in_dilations,
                                  const int64_t& in_group,
                                  TensorShapeVector& in_output_padding,
                                  TensorShapeVector& in_output_shape) {
    SetConvAttributes(auto_pad_str, kernel_shape, in_strides, pads_span,
                      in_dilations, in_group);

    output_padding = in_output_padding;
    output_shape = in_output_shape;
  }

  struct Prepare {
    TensorShape x_shape;
    TensorShape f_shape;
    TensorShape b_shape;
    TensorShape y_shape;
    int64_t N;
    int64_t num_input_channels;
    int64_t num_output_channels;
    TensorShape input_shape;
    TensorShapeVector kernel_shape;
    ConvPadVector pads;
    TensorShapeVector dilations;
    TensorShapeVector strides;
  };

  bool PrepareForCompute(Prepare& p, const TensorShape& p_shape,
                         const std::vector<int64_t> pads_data,
                         bool dynamic_padding = false,
                         const TensorShape* filter_shape = nullptr) const {
    const TensorShape& F_Shape =
        (filter_shape != nullptr) ? *filter_shape : p.f_shape;
    std::vector<int64_t> Pads;

    if (dynamic_padding) Pads = pads_data;

    TensorShape input_shape = p.x_shape.Slice(2);

    const int64_t num_input_channels = p.x_shape[1];
    const int64_t N = p.x_shape[0];
    const int64_t num_output_channels_multiplier = F_Shape[1];
    const int64_t num_output_channels = num_output_channels_multiplier * group;

    std::string message;
    // input validations

    if (group <= 0) {
      LOG(ERROR) << "group count is <= 0"
                 << " group: " << group;
      return false;
    }

    if (p.x_shape.NumDimensions() != F_Shape.NumDimensions()) {
      LOG(ERROR) << "X num_dims does not match W num_dims."
                 << " X: " << p.x_shape.ToString()
                 << " W: " << F_Shape.ToString();
      return false;
    }

    if (F_Shape[0] != num_input_channels) {
      LOG(ERROR) << "filter number not equal to input channel number."
                 << " filter_number: " << F_Shape[0]
                 << " num_input_channels: " << num_input_channels;
      return false;
    }

    // it looks like num_output_channels is really k*group similar to how in
    // the conv case num_input_channels is k*group. hence removing the check
    // for num_output_channels here.

    if (num_input_channels % group != 0) {
      LOG(ERROR) << "Input channels is not divisible by group."
                 << " num_input_channels: " << num_input_channels
                 << " group: " << group;
      return false;
    }

    TensorShapeVector kernel_shape;
    assert(ComputeKernelShape(F_Shape, kernel_shape));

    TensorShapeVector local_output_padding(output_padding);
    if (local_output_padding.empty()) {
      local_output_padding.resize(kernel_shape.size(), 0);
    }

    ConvPadVector local_pads;
    local_pads.reserve(2 * (input_shape.NumDimensions()));
    if (dynamic_padding) {
      for (int64_t i = 0; i < p_shape.SizeFromDimension(0); ++i) {
        local_pads.push_back(pads_data[i]);
      }
    } else {
      local_pads.assign(pads.begin(), pads.end());
    }

    if (local_pads.empty()) {
      local_pads.resize(kernel_shape.size() * 2, 0);
    }

    TensorShapeVector local_dilations(dilations);
    if (local_dilations.empty()) {
      local_dilations.resize(kernel_shape.size(), 1);
    }

    TensorShapeVector local_strides(strides);
    if (local_strides.empty()) {
      local_strides.resize(kernel_shape.size(), 1);
    }

    TensorShapeVector Y_dims;

    ComputePadsAndOutputShape(input_shape, num_output_channels, kernel_shape,
                              local_strides, local_dilations,
                              local_output_padding, N, &local_pads, &Y_dims);
    TensorShape Yshape(Y_dims);

    p.y_shape = Yshape;
    p.N = N;
    p.input_shape = std::move(input_shape);
    p.num_input_channels = num_input_channels;
    p.num_output_channels = num_output_channels;
    p.kernel_shape = std::move(kernel_shape);
    p.pads = std::move(local_pads);
    p.strides = std::move(local_strides);
    p.dilations = std::move(local_dilations);
    return true;
  }

  void ComputePadsAndOutputShape(TensorShape input_shape,
                                 int64_t output_channel,
                                 const TensorShapeVector& kernel_shape,
                                 const TensorShapeVector& p_strides,
                                 const TensorShapeVector& p_dilations,
                                 const TensorShapeVector& p_output_padding,
                                 const int64_t N, ConvPadVector* p_pads,
                                 TensorShapeVector* output_shape_p) const {
    size_t output_shape_size = output_shape.size();
    output_shape_p->insert(output_shape_p->begin(), {N, output_channel});

    size_t rank = input_shape.NumDimensions();
    for (size_t dim = 0; dim < rank; ++dim) {
      int64_t dim_size = -1;

      if (output_shape_size != 0) {
        dim_size = output_shape_size == rank ? output_shape[dim]
                                             : output_shape[dim + 2];
      }

      ComputeTransposePadAndOutputShape(
          input_shape[dim], p_strides[dim], kernel_shape[dim], p_dilations[dim],
          p_output_padding[dim], auto_pad, &p_pads->at(dim),
          &p_pads->at(input_shape.NumDimensions() + dim), &dim_size);

      ORT_ENFORCE(dim_size > 0, std::string("Invalid input shape: ") +
                                    input_shape.ToString());
      output_shape_p->push_back(dim_size);
    }
  }

  TensorShapeVector output_padding;
  TensorShapeVector output_shape;

 private:
  int64_t ComputeTotalPad(int64_t in_size, int64_t stride, int64_t adj,
                          int64_t kernel, int64_t dilation,
                          int64_t out_size) const {
    return std::max<int64_t>(0, (in_size - 1) * stride + adj +
                                    (kernel - 1) * dilation + 1 - out_size);
  }

  void DistributePadding(AutoPadType pad_type, const int64_t& total_pad,
                         int64_t& pad_head, int64_t& pad_tail) const {
    if (pad_type ==
        AutoPadType::SAME_LOWER) {  // pad more on head when total_pad is odd.
      pad_head = total_pad - total_pad / 2;
      pad_tail = total_pad / 2;
    } else {
      // for pad_type is NOTSET, SAME_LOWER or VALID
      // set pad_head as total_pad/2, pad_tail as total_pad-total_pad/2.
      // That said, we pad more on tail when total_pad is odd.
      pad_head = total_pad / 2;
      pad_tail = total_pad - total_pad / 2;
    }
  }

  void ComputeTransposePadAndOutputShape(
      const int64_t in_size, const int64_t stride, const int64_t kernel,
      const int64_t dilation, const int64_t adj, AutoPadType pad_type,
      int64_t* pad_head, int64_t* pad_tail, int64_t* out_size) const {
    // Output shape is explicitly provided - pad values will have to be
    // computed
    if (*out_size != -1) {
      assert(*out_size >= 0);
      // total pad
      auto total_pad =
          ComputeTotalPad(in_size, stride, adj, kernel, dilation, *out_size);
      DistributePadding(pad_type, total_pad, *pad_head, *pad_tail);
      return;
    }

    // Output shape is not provided - it needs to be computed along with pad
    // values (if applicable)

    // Compute padding if the auto_pad attribute is SAME_UPPER/SAME_LOWER
    if (pad_type == AutoPadType::SAME_UPPER ||
        pad_type == AutoPadType::SAME_LOWER) {
      // The ONNX spec says if `auto_pad` attribute is set, pad until the
      // `out_size` is `in_size * stride`
      auto total_pad = ComputeTotalPad(in_size, stride, adj, kernel, dilation,
                                       /*out_size = */ in_size * stride);
      DistributePadding(pad_type, total_pad, *pad_head, *pad_tail);
    }

    *out_size = (in_size - 1) * stride + adj + (kernel - 1) * dilation + 1 -
                *pad_head - *pad_tail;
  }
};

}  // namespace Cudnn

#endif  // CUDNN_COMMON_CONV_TRANSPOSE_ATTRIBUTES_HPP
