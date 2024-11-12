#include "gsl-lite.hpp"

#define NONE
#include <cassert>
#include <iostream>
#include <iterator>

#include "cudnn/common/tensor_shape.hpp"

namespace Cudnn {

template <class T, std::size_t N>
size_t size(const T (&a)[N]) noexcept {
  return N;
}

TensorShape::TensorShape(gsl::span<const int64_t> dims) {
  Allocate(dims.size());
  gsl::copy(dims, values_);
}

TensorShape& TensorShape::operator=(const TensorShape& other) {
  if (&other == this) return *this;

  Allocate(other.values_.size());
  gsl::copy(other.GetDims(), values_);
  return *this;
}

TensorShape& TensorShape::operator=(TensorShape&& other) noexcept {
  if (&other == this) return *this;

  // If the other TensorShape allocated a buffer, then take ownership of it
  if (other.allocated_buffer_) {
    allocated_buffer_ = std::move(other.allocated_buffer_);
    values_ = other.values_;
  } else {
    operator=(other);  // Otherwise we do a copy using the regular operator=
  }

  other.values_ = {};  // Just to be safe, set the other to be an empty shape
  return *this;
}

void TensorShape::Allocate(size_t size) {
  if (values_.size() == size) return;

  allocated_buffer_.reset();

  if (size > Cudnn::size(small_buffer_)) {
    allocated_buffer_ = std::make_unique<int64_t[]>(size);
    values_ = gsl::span<int64_t>(allocated_buffer_.get(), size);
  } else {
    values_ = gsl::span<int64_t>(small_buffer_, size);
  }
}

/**
 * Return the total number of elements. Returns 1 for an empty (rank 0)
 * TensorShape.
 */
int64_t TensorShape::Size() const {
  int64_t size = SizeHelper(0, values_.size());
  // should we cache the size? as multiple operation may be expensive.
  return size;
}

int64_t TensorShape::SizeToDimension(size_t dimension) const {
  const size_t num_dims = values_.size();
  assert(dimension <= num_dims);

  int64_t size = SizeHelper(0, dimension);
  return size;
}

int64_t TensorShape::SizeFromDimension(size_t dimension) const {
  const size_t num_dims = values_.size();
  assert(dimension <= num_dims);

  int64_t size = SizeHelper(dimension, num_dims);
  return size;
}

TensorShape TensorShape::Slice(size_t dimstart, size_t dimend) const {
  assert(dimstart <= dimend && dimend <= values_.size());
  return TensorShape(GetDims().subspan(dimstart, dimend - dimstart));
}

// output dimensions
std::string TensorShape::ToString() const {
  std::string result;

  result.append("{");
  bool first = true;
  for (auto dim : GetDims()) {
    if (!first) {
      result.append(",");
    }

    result.append(std::to_string(dim));
    first = false;
  }
  result.append("}");

  return result;
}

int64_t TensorShape::SizeHelper(size_t start, size_t end) const {
  // Must return 1 for an empty sequence
  int64_t size = 1;
  for (size_t i = start; i < end; i++) {
    if ((*this)[i] < 0) return -1;
    size *= (*this)[i];
  }
  return size;
}

// operator<< to nicely output to a stream
std::ostream& operator<<(std::ostream& out, const TensorShape& shape) {
  return (out << shape.ToString());
}

}  // namespace Cudnn
