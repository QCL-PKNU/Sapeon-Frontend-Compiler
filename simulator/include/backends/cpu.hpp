#ifndef BACKENDS_CPU_HPP
#define BACKENDS_CPU_HPP

#include <memory>
#include <string>

#include "backends/backend.hpp"
#include "enums/error.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "tl/expected.hpp"

class CpuBackend : public Backend {
 public:
  static std::unique_ptr<Backend> CreateBackend();
};

#endif  // BACKENDS_CPU_HPP
