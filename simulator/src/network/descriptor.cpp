#include "network/descriptor.hpp"

#define SCOPE Descriptor

SCOPE::Descriptor() {}

int SCOPE::padding_height_top() { return padding_height_top_; }

void SCOPE::padding_height_top(int value) { padding_height_top_ = value; }

int SCOPE::padding_height_bottom() { return padding_height_bottom_; }

void SCOPE::padding_height_bottom(int value) { padding_height_bottom_ = value; }

int SCOPE::padding_width_left() { return padding_width_left_; }

void SCOPE::padding_width_left(int value) { padding_width_left_ = value; }

int SCOPE::padding_width_right() { return padding_width_right_; }

void SCOPE::padding_width_right(int value) { padding_width_right_ = value; }

int SCOPE::stride_height() { return stride_height_; }

void SCOPE::stride_height(int value) { stride_height_ = value; }

int SCOPE::stride_width() { return stride_width_; }

void SCOPE::stride_width(int value) { stride_width_ = value; }

int SCOPE::dilation_height() { return dilation_height_; }

void SCOPE::dilation_height(int value) { dilation_height_ = value; }

int SCOPE::dilation_width() { return dilation_width_; }

void SCOPE::dilation_width(int value) { dilation_width_ = value; }

int SCOPE::window_height() { return window_height_; }

void SCOPE::window_height(int value) { window_height_ = value; }

int SCOPE::window_width() { return window_width_; }

void SCOPE::window_width(int value) { window_width_ = value; }

int SCOPE::groups() { return groups_; }

void SCOPE::groups(int value) { groups_ = value; }

float SCOPE::scale() { return scale_; }

void SCOPE::scale(float value) { scale_ = value; }
