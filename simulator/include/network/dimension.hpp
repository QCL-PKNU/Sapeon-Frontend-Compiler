#ifndef NETWORK_DIMENSION_HPP
#define NETWORK_DIMENSION_HPP

#include <string>
#include <vector>

class Dimension {
 public:
  Dimension();
  Dimension(size_t n, size_t c, size_t h, size_t w);
  size_t n() const;
  void n(size_t value);
  size_t c() const;
  void c(size_t value);
  size_t h() const;
  void h(size_t value);
  size_t w() const;
  void w(size_t value);
  std::string str() const;
  size_t size() const;

  void dims(std::vector<size_t> dims);
  std::vector<size_t> dims() const;

 private:
  enum class DimType { DIM_NORMAL = 0, DIM_CUSTOM = 1 };
  size_t n_;
  size_t c_;
  size_t h_;
  size_t w_;

  std::vector<size_t> dims_;
  DimType type_;
};

#endif  // NETWORK_DIMENSION_HPP
