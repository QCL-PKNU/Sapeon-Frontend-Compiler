#include "image.hpp"

#include <fstream>
using std::ifstream;
#include <iostream>
#include <optional>
using std::optional;
#include <sstream>
#include <string>
using std::string;
using std::stringstream;
#include <vector>
using std::vector;
#include "glog/logging.h"
#include "json.hpp"
using nlohmann::json;

#ifdef OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#else
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif

namespace image {
Image::Image()
    : pre_scales_(3),
      pre_gammas_(3),
      pre_betas_(3),
      pre_biases_(3),
      pre_channels_(3) {}

void Image::CopyData(float *data, int size) {
  assert(size == w_ * h_ * c_);
  for (int i = 0; i < size; i++) {
    data[i] = data_[i];
  }
}

void Image::LoadImageColor(const std::string &filename) {
  LoadImage(filename, 3);
}

void Image::LoadImage(const std::string &filename, int c) {
#ifdef OPENCV
  LoadImageCv(filename, c);
#else
  LoadImageStb(filename, c);
#endif
}

#ifdef OPENCV
void Image::LoadImageCv(const std::string &filename, int channels) {
  int option;
  if (channels == 3) {
    option = 1;
  } else if (channels == 1) {
    option = 0;
  } else if (channels == 0) {
    option = -1;
  } else {
    LOG(ERROR) << "OpenCV can't force load with " << channels << " channels\n";
  }

  cv::Mat img = cv::imread(filename, option);
  if (!img.data) {
    LOG(ERROR) << "Cannot load image " << filename;
    exit(0);
  }

  cv::Mat rgb_img;
  cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);

  int w = rgb_img.size().width;
  int h = rgb_img.size().height;
  int c = rgb_img.channels();

  if (channels) {
    c = channels;
  }

  unsigned char *image_data = rgb_img.data;

  SetEmptyImage(w, h, c);

  for (int k = 0; k < c; ++k) {
    for (int j = 0; j < h; ++j) {
      for (int i = 0; i < w; ++i) {
        int dst_index = i + w * j + w * h * k;
        int src_index = k + c * i + c * w * j;
        data_[dst_index] = (float)image_data[src_index] / 255.;
      }
    }
  }
}

#else
void Image::LoadImageStb(const std::string &filename, int channels) {
  int w, h, c;
  unsigned char *image_data = stbi_load(filename.c_str(), &w, &h, &c, channels);
  if (!image_data) {
    LOG(ERROR) << "Cannot load image " << filename
               << "STB Reason: " << stbi_failure_reason();
    exit(0);
  }

  if (channels) {
    c = channels;
  }

  SetEmptyImage(w, h, c);

  for (int k = 0; k < c; ++k) {
    for (int j = 0; j < h; ++j) {
      for (int i = 0; i < w; ++i) {
        int dst_index = i + w * j + w * h * k;
        int src_index = k + c * i + c * w * j;
        data_[dst_index] = (float)image_data[src_index];
      }
    }
  }
  free(image_data);
}
#endif

void Image::SetEmptyImage(int w, int h, int c) {
  w_ = w;
  h_ = h;
  c_ = c;
  data_ = vector<float>(w * h * c);
}

void Image::ParsePreprocessParameter(const optional<string> &config_file_path) {
  if (!config_file_path.has_value()) {
    resize_size_ = std::nullopt;
    crop_size_ = std::nullopt;
    for (int i = 0; i < 3; ++i) {
      pre_scales_.at(i) = 1.0f;
      pre_gammas_.at(i) = 0.0f;
      pre_betas_.at(i) = 1.0f;
      pre_biases_.at(i) = 0.0f;
      pre_channels_.at(i) = i;
    }
    return;
  }

  ifstream config_file(config_file_path.value());
  json config;

  try {
    config = json::parse(config_file);

    if (config.contains("resize_size")) {
      resize_size_ = config["resize_size"];
    } else {
      resize_size_ = std::nullopt;
    }

    if (config.contains("crop_size")) {
      crop_size_ = config["crop_size"];
    } else {
      crop_size_ = std::nullopt;
    }

    if (config.contains("normalization")) {
      pre_scales_ = config["normalization"]["scale"].get<std::vector<float>>();
      pre_gammas_ = config["normalization"]["gamma"].get<std::vector<float>>();
      pre_biases_ = config["normalization"]["bias"].get<std::vector<float>>();
      pre_betas_ = config["normalization"]["beta"].get<std::vector<float>>();
      pre_channels_ =
          config["normalization"]["channel"].get<std::vector<int>>();
    } else {
      pre_scales_ = std::vector<float>{1.0f, 1.0f, 1.0f};
      pre_gammas_ = std::vector<float>{0.0f, 0.0f, 0.0f};
      pre_betas_ = std::vector<float>{1.0f, 1.0f, 1.0f};
      pre_biases_ = std::vector<float>{0.0f, 0.0f, 0.0f};
      pre_channels_ = std::vector<int>{0, 1, 2};
    }
  } catch (const std::runtime_error &e) {
    LOG(ERROR) << e.what();
  }
}

void Image::ConvertFormat() {
  int size = w_ * h_;
  vector<float> orig_data = vector<float>(data_.begin(), data_.end());
  for (int idx = 0; idx < size; ++idx) {
    for (int k = 0; k < 3; ++k) {
      auto temp =
          orig_data[idx + size * k] * (pre_scales_[k] / 255.F) - pre_gammas_[k];
      data_[idx + size * pre_channels_[k]] =
          (temp / pre_betas_[k]) + pre_biases_[k];
    }
  }
}

void Image::ResizeImage(int w, int h) {
  int orig_w = w_;
  int orig_h = h_;
  int orig_c = c_;
  auto data_part = vector<float>(w * orig_h * orig_c);
  float w_scale = (float)(orig_w - 1) / (w - 1);
  float h_scale = (float)(orig_h - 1) / (h - 1);

  for (int k = 0; k < orig_c; ++k) {
    for (int j = 0; j < orig_h; ++j) {
      for (int i = 0; i < w; ++i) {
        float val = 0;
        if (i == w - 1 || orig_w == 1) {
          val = GetPixel(orig_w - 1, j, k);
        } else {
          float sx = i * w_scale;
          int ix = (int)sx;
          float dx = sx - ix;
          val = (1 - dx) * GetPixel(ix, j, k) + dx * GetPixel(ix + 1, j, k);
        }
        data_part[k * orig_h * w + j * w + i] = val;
      }
    }
  }

  SetEmptyImage(w, h, orig_c);

  for (int c = 0; c < orig_c; ++c) {
    for (int y = 0; y < h; ++y) {
      float sy = y * h_scale;
      int iy = (int)sy;
      float dy = sy - iy;
      for (int x = 0; x < w; ++x) {
        float pixel_val = data_part[c * orig_h * w + iy * w + x];
        float val = (1 - dy) * pixel_val;
        SetPixel(x, y, c, val);
      }
      if (y == h - 1 || orig_h == 1) continue;
      for (int x = 0; x < w; ++x) {
        float pixel_val = data_part[c * orig_h * w + (iy + 1) * w + x];
        float val = dy * pixel_val;
        AddPixel(x, y, c, val);
      }
    }
  }
}

void Image::CropImage(int dx, int dy, int w, int h) {
  int orig_w = w_;
  int orig_h = h_;
  int orig_c = c_;
  auto orig_data = data_;
  SetEmptyImage(w, h, orig_c);
  for (int k = 0; k < orig_c; ++k) {
    for (int j = 0; j < h; ++j) {
      for (int i = 0; i < w; ++i) {
        int y = j + dy;
        int x = i + dx;
        y = Clip(y, 0, orig_h - 1);
        x = Clip(x, 0, orig_w - 1);
        float cropped_value = orig_data[k * orig_h * orig_w + y * orig_w + x];
        SetPixel(i, j, k, cropped_value);
      }
    }
  }
}

void Image::CenterCropImage(int w, int h) {
  int orig_w = w_;
  int orig_h = h_;
  int dx = static_cast<int>(orig_w / 2.0 - w / 2.0);
  int dy = static_cast<int>(orig_h / 2.0 - h / 2.0);
  CropImage(dx, dy, w, h);
}

void Image::SquareImage() {
  int orig_w = w_;
  int orig_h = h_;
  int m = (orig_w < orig_h) ? orig_w : orig_h;
  CropImage((orig_w - m) / 2, (orig_h - m) / 2, m, m);
}

float Image::GetPixel(int x, int y, int c) {
  assert(x < w_ && y < h_ && c < c_);
  return data_[c * h_ * w_ + y * w_ + x];
}

void Image::SetPixel(int x, int y, int c, float val) {
  if (x < 0 || y < 0 || c < 0 || x >= w_ || y >= h_ || c >= c_) return;
  assert(x < w_ && y < h_ && c < c_);
  data_[c * h_ * w_ + y * w_ + x] = val;
}

void Image::AddPixel(int x, int y, int c, float val) {
  assert(x < w_ && y < h_ && c < c_);
  data_[c * h_ * w_ + y * w_ + x] += val;
}

int Image::Clip(int a, int min, int max) {
  if (a < min) {
    return min;
  }
  if (a > max) {
    return max;
  }
  return a;
}

void Image::DumpToBinary(const std::string &file_path) {
  std::ofstream dump;
  dump.open(file_path, std::ios::binary);
  dump.write(reinterpret_cast<const char *>(&data_[0]),
             sizeof(float) * data_.size());
  dump.close();
}

vector<float> &Image::pre_scales() { return pre_scales_; }

vector<float> &Image::pre_gammas() { return pre_gammas_; }

vector<float> &Image::pre_betas() { return pre_betas_; }

vector<float> &Image::pre_biases() { return pre_biases_; }

vector<int> &Image::pre_channels() { return pre_channels_; }

std::optional<int> Image::resize_size() { return resize_size_; }

std::optional<int> Image::crop_size() { return crop_size_; }
}  // namespace image
