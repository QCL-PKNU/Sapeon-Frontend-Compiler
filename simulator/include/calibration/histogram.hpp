#ifndef CALIBRATION_HISTOGRAM_HPP
#define CALIBRATION_HISTOGRAM_HPP

#include <memory>
#include <mutex>
#include <vector>

#include "network/tensor.hpp"

namespace spgraph_simulator {
namespace calibration {
class Histogram {
 public:
  Histogram(size_t num_histogram_bins);
  void SetRange(float range);
  void SetRange(const Tensor &tensor);
  void Collect(const Tensor &tensor);
  size_t TotalCount() const noexcept;
  size_t num_bins() const { return num_bins_; }
  size_t histogram(int idx) const { return histogram_.at(idx); }
  const std::vector<size_t> &histogram() const { return histogram_; }
  float histogram_edge(int idx) const { return histogram_edge_.at(idx); }
  const std::vector<float> &histogram_edge() const { return histogram_edge_; }

 private:
  size_t Stack(float value) noexcept;
  bool IsHistogramInitialized() noexcept;
  size_t num_bins_;
  std::vector<size_t> histogram_;
  std::vector<float> histogram_edge_;
};
}  // namespace calibration
}  // namespace spgraph_simulator

#endif  // CALIBRATION_HISTOGRAM_HPP
