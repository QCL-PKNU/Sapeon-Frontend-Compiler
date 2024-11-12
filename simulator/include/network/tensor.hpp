#ifndef NETWORK_TENSOR_HPP
#define NETWORK_TENSOR_HPP

#include <vector>

#include "datatype.hpp"
#include "network/dimension.hpp"

class Tensor {
 public:
  Tensor(size_t n, size_t c, size_t h, size_t w, dty::DataType dtype);
  Tensor(std::vector<size_t> dims, dty::DataType dtype);
  Tensor(size_t size, dty::DataType dtype);
  Tensor(const Tensor &other);
  Tensor &operator=(const Tensor &other);
  Tensor(Tensor &&other) noexcept;
  Tensor &operator=(Tensor &&other) noexcept;
  ~Tensor();
  void *data();
  const void *data() const;

  template <typename Type>
  Type *data() {
    return static_cast<Type *>(data_);
  };
  template <typename Type>
  const Type *data() const {
    return static_cast<Type *>(data_);
  };

  const Dimension &dimension() const;
  size_t n() const;
  size_t c() const;
  size_t h() const;
  size_t w() const;
  dty::DataType dtype() const;
  size_t size() const;

 private:
  void dimension(size_t n, size_t c, size_t h, size_t w);
  void dimension(std::vector<size_t> dims);
  Dimension dimension_;
  void *data_;
  dty::DataType dtype_;
};

#endif  // NETWORK_TENSOR_HPP
