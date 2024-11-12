#ifndef NETWORK_DESCRIPTOR_HPP
#define NETWORK_DESCRIPTOR_HPP

class Descriptor {
 public:
  Descriptor();
  int padding_height_top();
  void padding_height_top(int value);
  int padding_height_bottom();
  void padding_height_bottom(int value);
  int padding_width_left();
  void padding_width_left(int value);
  int padding_width_right();
  void padding_width_right(int value);
  int stride_height();
  void stride_height(int value);
  int stride_width();
  void stride_width(int value);
  int dilation_height();
  void dilation_height(int value);
  int dilation_width();
  void dilation_width(int value);
  int window_height();
  void window_height(int value);
  int window_width();
  void window_width(int value);
  int groups();
  void groups(int value);
  float scale();
  void scale(float value);

 private:
  int padding_height_top_;
  int padding_height_bottom_;
  int padding_width_left_;
  int padding_width_right_;
  int stride_height_;
  int stride_width_;
  int dilation_height_;
  int dilation_width_;
  int window_height_;
  int window_width_;
  int groups_;
  float scale_;
};

#endif  // NETWORK_DESCRIPTOR_HPP
