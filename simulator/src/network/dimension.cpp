#include "network/dimension.hpp"

#define SCOPE Dimension

#include <string>
#include <vector>

SCOPE::Dimension() : n_(0), c_(0), h_(0), w_(0) { type_ = DimType::DIM_NORMAL; }

SCOPE::Dimension(size_t n, size_t c, size_t h, size_t w)
    : n_(n), c_(c), h_(h), w_(w) {
  type_ = DimType::DIM_NORMAL;
}

size_t SCOPE::n() const { return n_; }

void SCOPE::n(size_t value) { n_ = value; }

size_t SCOPE::c() const { return c_; }

void SCOPE::c(size_t value) { c_ = value; }

size_t SCOPE::h() const { return h_; }

void SCOPE::h(size_t value) { h_ = value; }

size_t SCOPE::w() const { return w_; }

void SCOPE::w(size_t value) { w_ = value; }

void SCOPE::dims(std::vector<size_t> dims) {
  dims_ = dims;
  if (dims_.size() == 4) {
    n_ = dims[0];
    c_ = dims[1];
    h_ = dims[2];
    w_ = dims[3];
  }

  type_ = DimType::DIM_CUSTOM;
}

std::vector<size_t> SCOPE::dims() const {
  if (type_ == DimType::DIM_NORMAL) {
    std::vector<size_t> dims = {n_, c_, h_, w_};
    return dims;
  } else {
    return dims_;
  }
}

std::string SCOPE::str() const {
  std::string dimension = "";

  if (n_ != 0)
    dimension = std::to_string(n_) + " x " + std::to_string(c_) + " x " +
                std::to_string(h_) + " x " + std::to_string(w_);

  return dimension;
}

size_t SCOPE::size() const {
  if (type_ == DimType::DIM_NORMAL) {
    return n_ * c_ * h_ * w_;
  } else {
    if (dims_.size() == 0) {
      return 0;
    } else {
      size_t size = 1;
      for (size_t index = 0; index < dims_.size(); index++) {
        size *= dims_[index];
      }

      return size;
    }
  }
}
