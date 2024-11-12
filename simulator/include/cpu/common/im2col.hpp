#ifndef CPU_COMMON_IM2COL_HPP
#define CPU_COMMON_IM2COL_HPP

#include <cstddef>
#include <cstdint>

namespace cpu {

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
};

// From Berkeley Vision's Caffe!
// https://github.com/BVLC/caffe/blob/master/LICENSE
template <typename Type>
void im2col_cpu(const Type* data_im, int channels, int height, int width,
                int kernel_h, int kernel_w, int pad_ht, int pad_hb, int pad_wl,
                int pad_wr, int stride_h, int stride_w, int dilation_h,
                int dilation_w, Type* data_col) {
  const int output_h =
      (height + (pad_ht + pad_hb) - (dilation_h * (kernel_h - 1) + 1)) /
          stride_h +
      1;
  const int output_w =
      (width + (pad_wl + pad_wr) - (dilation_w * (kernel_w - 1) + 1)) /
          stride_w +
      1;
  const int channel_size = height * width;
  for (int channel = channels; channel > 0; channel--) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_ht + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_col = output_w; output_col; output_col--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_wl + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
    data_im += channel_size;
  }
}

template <typename Type>
void im2col_mxc(const Type* data_im, int channels, int height, int width,
                int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h,
                int stride_w, int dilation_h, int dilation_w, Type* data_col) {
  const int output_h =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  int channel, kernel_row, kernel_col, output_rows, output_col;
  for (channel = channels; channel--; data_im += channel_size) {
    for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (output_col = output_w; output_col; output_col--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}
}  // namespace cpu

#endif  // CPU_COMMON_IM2COL_HPP
