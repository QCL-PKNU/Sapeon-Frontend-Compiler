#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <optional>
#include <string>
#include <vector>

namespace image {
class Image {
 public:
  Image();
  void LoadImageColor(const std::string &filename);
  void ConvertFormat();
  void ResizeImage(int size);
  void ResizeImage(int w, int h);
  void CenterCropImage(int w, int h);
  void SquareImage();
  void CopyData(float *data, int size);
  void ParsePreprocessParameter(
      const std::optional<std::string> &config_file_path);
  void DumpToBinary(const std::string &file_path);
  std::vector<float> &pre_scales();
  std::vector<float> &pre_gammas();
  std::vector<float> &pre_betas();
  std::vector<float> &pre_biases();
  std::vector<int> &pre_channels();
  std::optional<int> resize_size();
  std::optional<int> crop_size();

 private:
  void LoadImage(const std::string &filename, int channels);
#ifdef OPENCV
  void LoadImageCv(const std::string &filename, int channels);
#else
  void LoadImageStb(const std::string &filename, int channels);
#endif
  void CropImage(int dx, int dy, int w, int h);
  float GetPixel(int x, int y, int c);
  void SetPixel(int x, int y, int c, float val);
  void AddPixel(int x, int y, int c, float val);
  int Clip(int a, int min, int max);
  void SetEmptyImage(int w, int h, int c);
  int w_;
  int h_;
  int c_;
  std::optional<int> resize_size_;
  std::optional<int> crop_size_;
  std::vector<float> pre_scales_;
  std::vector<float> pre_gammas_;
  std::vector<float> pre_betas_;
  std::vector<float> pre_biases_;
  std::vector<int> pre_channels_;
  std::vector<float> data_;
};
}  // namespace image

#endif  // IMAGE_HPP
