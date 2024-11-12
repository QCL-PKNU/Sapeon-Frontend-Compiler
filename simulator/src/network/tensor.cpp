#include "network/tensor.hpp"

#include <cstdlib>
#include <cstring>
#include <vector>

#include "datatype.hpp"
#include "glog/logging.h"
#include "network/dimension.hpp"

Tensor::Tensor(size_t n, size_t c, size_t h, size_t w, dty::DataType dtype) {
  dimension(n, c, h, w);
  dtype_ = dtype;
  data_ = std::malloc(this->size());
}

Tensor::Tensor(std::vector<size_t> dims, dty::DataType dtype) {
  dimension(dims);
  dtype_ = dtype;
  data_ = std::malloc(this->size());
}

Tensor::Tensor(size_t size, dty::DataType dtype) {
  dimension(1, size, 1, 1);
  dtype_ = dtype;
  data_ = std::malloc(this->size());
}

Tensor::Tensor(const Tensor& other) {
  dimension(other.dimension().dims());
  dtype_ = other.dtype();
  data_ = std::malloc(this->size());
  std::memcpy(data_, other.data_, this->size());
}

Tensor& Tensor::operator=(const Tensor& other) {
  if (this != &other) {
    dimension(other.dimension().dims());
    dtype_ = other.dtype();

    void* new_ptr = nullptr;
    if (other.data_ != nullptr) {
      new_ptr = std::malloc(this->size());
      std::memcpy(data_, other.data_, this->size());
    }
    std::free(data_);
    data_ = new_ptr;
  }
  return *this;
}

Tensor::Tensor(Tensor&& other) noexcept : data_(nullptr) {
  dimension_ = std::move(other.dimension_);
  dtype_ = other.dtype_;
  data_ = other.data_;
  other.data_ = nullptr;
};

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    std::free(data_);
    dimension_ = std::move(other.dimension_);
    dtype_ = other.dtype_;
    data_ = other.data_;
    other.data_ = nullptr;
  }
  return *this;
}

Tensor::~Tensor() { std::free(data_); }

const Dimension& Tensor::dimension() const { return dimension_; }

void Tensor::dimension(size_t n, size_t c, size_t h, size_t w) {
  dimension_.n(n);
  dimension_.c(c);
  dimension_.h(h);
  dimension_.w(w);
}

void Tensor::dimension(std::vector<size_t> dims) {
  dimension_.dims(dims);
  if (dimension_.dims().size() == 4) {
    dimension_.n(dims[0]);
    dimension_.c(dims[1]);
    dimension_.h(dims[2]);
    dimension_.w(dims[3]);
  }
}

void* Tensor::data() { return data_; }

const void* Tensor::data() const { return data_; }

size_t Tensor::n() const { return dimension_.n(); }

size_t Tensor::c() const { return dimension_.c(); }

size_t Tensor::h() const { return dimension_.h(); }

size_t Tensor::w() const { return dimension_.w(); }

dty::DataType Tensor::dtype() const { return dtype_; }

size_t Tensor::size() const { return dimension_.size() * dty::SizeOf(dtype_); }
