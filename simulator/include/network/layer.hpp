#ifndef NETWORK_LAYER_HPP
#define NETWORK_LAYER_HPP

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "datatype.hpp"
#include "factory.hpp"
#include "network/descriptor.hpp"
#include "network/dimension.hpp"
#include "network/tensor.hpp"
#include "x220/quant_config.hpp"
#include "x330/quant_config.hpp"

class Layer {
 public:
  Layer();

  bool operator<(const Layer rhs) const { return this->id_ < rhs.id_; }

  bool CheckSupportedOperation(const std::string &operation_name,
                               const std::string &backend_type);
  bool CheckSupportedActivation(const std::string &activation_name,
                                const std::string &backend_type);
  bool CheckValidLayer(const std::string &backend_type, bool do_quant);

  void id(int id);
  int id();

  void name(std::string name);
  std::string name();

  void type(const std::string &type);
  const std::string &type() const;

  void predecessors(std::vector<int> predecessors);
  std::vector<int> &predecessors();

  int predecessors(int idx);
  void successors(std::vector<int> successors);

  std::vector<int> &successors();
  int successors(int idx);

  std::vector<std::shared_ptr<Tensor>> &inputs() { return inputs_; }
  std::shared_ptr<Tensor> inputs(int idx) { return inputs_.at(idx); }
  void inputs(const std::vector<Tensor> &activations);

  std::vector<Dimension> &input_dimensions();
  void input_dimensions(std::vector<Dimension> dimension);
  Dimension &input_dimensions(int idx);

  const Dimension &output_dimension() const;
  void output_dimension(const Dimension &dimension);
  void output_dimension(Dimension &&dimension);

  std::shared_ptr<Tensor> intermediate_activation();
  void intermediate_activation(std::shared_ptr<Tensor> act);

  void operation_types(const std::vector<std::string> &operation_types);
  const std::vector<std::string> &operation_types() const;
  const std::string &operation_types(int) const;
  bool HasOperationTypes(const std::string &ops_type);

  std::shared_ptr<Tensor> filter() const;
  void filter(std::shared_ptr<Tensor> data);
  bool HasFilter() const;

  std::shared_ptr<Tensor> bias();
  void bias(std::shared_ptr<Tensor> data);
  bool HasBias();

  std::shared_ptr<Tensor> scale();
  void scale(std::shared_ptr<Tensor> data);
  bool HasScale();

  std::shared_ptr<Tensor> mean();
  void mean(std::shared_ptr<Tensor> data);
  bool HasMean();

  std::shared_ptr<Tensor> variance();
  void variance(std::shared_ptr<Tensor> data);
  bool HasVariance();

  std::shared_ptr<Descriptor> convolution();
  void convolution(std::shared_ptr<Descriptor> desc);
  bool HasConvolutionDescriptor();

  std::shared_ptr<Descriptor> sampling();
  void sampling(std::shared_ptr<Descriptor> desc);
  bool HasSamplingDescriptor();

  std::shared_ptr<Descriptor> ewadd();
  void ewadd(std::shared_ptr<Descriptor> desc);
  bool HasEwaddDescriptor();

  std::shared_ptr<Descriptor> ewmul();
  void ewmul(std::shared_ptr<Descriptor> desc);
  bool HasEwmulDescriptor();

  std::string &activation_type();
  void activation_type(const std::string &);
  bool HasActivation();

  float epsilon();
  void epsilon(float value);

  float alpha();
  void alpha(float value);

  float gamma();
  void gamma(float value);

  int axis();
  void axis(int value);

  int stash_type();
  void stash_type(int value);

  float beta();
  void beta(float value);

  int64_t trans_A();
  void trans_A(int64_t value);

  int64_t trans_B();
  void trans_B(int64_t value);

  int64_t keepdims();
  void keepdims(int64_t value);

  int64_t noop_with_empty_axes();
  void noop_with_empty_axes(int64_t value);

  int64_t select_last_index();
  void select_last_index(int64_t value);

  std::vector<int64_t> &axes();
  void axes(std::vector<int64_t> value);

  std::string mode() const;
  void mode(const std::string value);

  float extrapolation_value();
  void extrapolation_value(float value);

  std::string coordinate_transformation_mode() const;
  void coordinate_transformation_mode(const std::string value);

  std::string nearest_mode() const;
  void nearest_mode(const std::string value);

  float cubic_coeff_a();
  void cubic_coeff_a(float value);

  int64_t exclude_outside();
  void exclude_outside(int64_t value);

  std::string auto_pad() const;
  void auto_pad(const std::string value);

  int64_t group();
  void group(int64_t value);

  std::vector<int64_t> &dilations();
  void dilations(std::vector<int64_t> value);

  std::vector<int64_t> &kernel_shape();
  void kernel_shape(std::vector<int64_t> value);

  std::vector<int64_t> &output_padding();
  void output_padding(std::vector<int64_t> value);

  std::vector<int64_t> &output_shape();
  void output_shape(std::vector<int64_t> value);

  std::vector<int64_t> &pads();
  void pads(std::vector<int64_t> value);

  std::vector<int64_t> &strides();
  void strides(std::vector<int64_t> value);

  std::vector<float> &input_thresholds();
  float input_thresholds(int idx);
  void input_thresholds(std::vector<float> thresholds);

  float output_threshold();
  void output_threshold(float threshold);

  std::vector<float> &filter_thresholds();
  void filter_thresholds(std::vector<float> thresholds);

  x220::QuantConfig &x220_quant_config();
  void x220_quant_config(std::shared_ptr<x220::QuantConfig> quant_config);

  x330::QuantConfig &x330_quant_config();
  void x330_quant_config(std::shared_ptr<x330::QuantConfig> quant_config);

  std::vector<float> &negative_slope();
  float negative_slope(int idx);
  void negative_slope(const std::vector<float> &negative_slope);

 private:
  int id_;
  std::string name_;
  std::string type_;
  std::vector<int> predecessors_;
  std::vector<int> successors_;
  std::vector<std::string> operation_types_;
  std::vector<std::shared_ptr<Tensor>> inputs_;
  std::vector<Dimension> input_dimensions_;
  Dimension output_dimension_;
  std::shared_ptr<Tensor> intermediate_activation_;
  std::shared_ptr<Tensor> filter_;
  std::shared_ptr<Tensor> bias_;
  std::shared_ptr<Tensor> scale_;
  std::shared_ptr<Tensor> mean_;
  std::shared_ptr<Tensor> variance_;

  std::shared_ptr<Descriptor> convolution_;
  std::shared_ptr<Descriptor> sampling_;
  std::shared_ptr<Descriptor> element_wise_add_;
  std::shared_ptr<Descriptor> element_wise_mul_;
  std::string activation_type_;
  float epsilon_;
  float alpha_;
  float gamma_;

  float axis_;
  int stash_type_;

  float beta_;
  int64_t trans_A_;
  int64_t trans_B_;

  std::vector<int64_t> axes_;
  int64_t keepdims_;
  int64_t noop_with_empty_axes_;
  int64_t select_last_index_;

  std::string mode_;
  float extrapolation_value_;
  std::string coordinate_transformation_mode_;
  std::string nearest_mode_;
  float cubic_coeff_a_;
  int64_t exclude_outside_;

  std::string auto_pad_;
  std::vector<int64_t> dilations_;
  int64_t group_;
  std::vector<int64_t> kernel_shape_;
  std::vector<int64_t> output_padding_;
  std::vector<int64_t> output_shape_;
  std::vector<int64_t> pads_;
  std::vector<int64_t> strides_;

  std::vector<float> input_thresholds_;
  float output_threshold_;
  std::vector<float> filter_thresholds_;

  std::vector<float> neg_slopes_;

  std::shared_ptr<x220::QuantConfig> x220_quant_config_;
  std::shared_ptr<x330::QuantConfig> x330_quant_config_;
};

#endif  // NETWORK_LAYER_HPP
