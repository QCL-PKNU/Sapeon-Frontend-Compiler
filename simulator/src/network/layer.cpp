#include "network/layer.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "factory.hpp"
#include "glog/logging.h"
#include "network/descriptor.hpp"
#include "network/dimension.hpp"
#include "operations/cpu_operation.hpp"
#ifdef GPU
#include "operations/cudnn_operation.hpp"
#endif
#include "x220/quant_config.hpp"

Layer::Layer()
    : intermediate_activation_(nullptr),
      filter_(nullptr),
      bias_(nullptr),
      scale_(nullptr),
      mean_(nullptr),
      variance_(nullptr),
      convolution_(nullptr),
      sampling_(nullptr),
      element_wise_add_(nullptr),
      activation_type_(""),
      epsilon_(1e-5),
      alpha_(std::numeric_limits<float>::quiet_NaN()),
      gamma_(std::numeric_limits<float>::quiet_NaN()),
      axis_(std::numeric_limits<float>::quiet_NaN()),
      stash_type_(std::numeric_limits<int>::lowest()),
      beta_(std::numeric_limits<float>::quiet_NaN()),
      trans_A_(std::numeric_limits<int64_t>::lowest()),
      trans_B_(std::numeric_limits<int64_t>::lowest()),
      keepdims_(std::numeric_limits<int64_t>::lowest()),
      noop_with_empty_axes_(std::numeric_limits<int64_t>::lowest()),
      select_last_index_(std::numeric_limits<int64_t>::lowest()),
      extrapolation_value_(std::numeric_limits<float>::quiet_NaN()),
      coordinate_transformation_mode_(""),
      nearest_mode_(""),
      cubic_coeff_a_(std::numeric_limits<float>::quiet_NaN()),
      exclude_outside_(std::numeric_limits<int64_t>::lowest()),
      auto_pad_(""),
      group_(std::numeric_limits<int64_t>::lowest()),
      output_threshold_(-1.0f),
      name_({}),
      x220_quant_config_(std::make_shared<x220::QuantConfig>()),
      type_({}) {}

bool Layer::CheckSupportedOperation(const std::string &operation_name,
                                    const std::string &backend_type) {
  if (operation_name == "Activations") {
    return CheckSupportedActivation(activation_type_, backend_type);
  }
  if (backend_type == "cpu") {
    auto p_operation = Factory<CpuOperation>::CreateInstance(operation_name);
    LOG(INFO) << "C========>  " << operation_name;
    if (p_operation == nullptr) {
      DLOG(ERROR) << "Failed to create CpuOperation: " << operation_name;
      return false;
    }
  }
#ifdef GPU
  else if (backend_type == "cudnn") {
    auto p_operation =
        Factory<CudnnOperation<float>>::CreateInstance(operation_name);
    if (p_operation == nullptr) {
      DLOG(ERROR) << "Failed to create CudnnOperation: " << operation_name;
      return false;
    }
  }
#endif
  return true;
}

bool Layer::CheckSupportedActivation(const std::string &activation_name,
                                     const std::string &backend_type) {
  if (backend_type == "cpu") {
    auto p_operation = Factory<CpuOperation>::CreateInstance(activation_name);
    if (p_operation == nullptr) {
      DLOG(ERROR) << "Failed to create CpuOperation: " << activation_name;
      return false;
    }
  }
#ifdef GPU
  else if (backend_type == "cudnn") {
    auto p_operation =
        Factory<CudnnOperation<float>>::CreateInstance(activation_name);
    if (p_operation == nullptr) {
      DLOG(ERROR) << "Failed to create CudnnOperation: " << activation_name;
      return false;
    }
  }
#endif
  return true;
}

bool Layer::CheckValidLayer(const std::string &backend_type, bool do_quant) {
  Dimension dims = input_dimensions_.at(0);

  LOG(INFO) << "-------------------------------------------------------------------------------- " << id_;

  for (auto operation_name : operation_types_) {
    if (!CheckSupportedOperation(operation_name, backend_type)) return false;
    
    if (backend_type == "cpu") {
      if (id_ < 172) {
        auto p_operation = Factory<CpuOperation>::CreateInstance(operation_name);
        p_operation->CheckValidOperation(*this, dims);
        dims = p_operation->CalculateOutputDimension(*this, dims);

        LOG(INFO) << "Layer " << id_ << ": "
                << "expected output dimension = [" << output_dimension_.str()
                << "], actual output dimension = [" << dims.str() << "]";
      } 
      // else {
      //   LOG(INFO) << "Op " << operation_name;

      //   if (operation_name == "SkipConvolution") {
      //     auto p_operation = Factory<CpuOperation>::CreateInstance(operation_name);
      //     p_operation->CheckValidOperation(*this, dims);
      //     dims = p_operation->CalculateOutputDimension(*this, dims);

      //     LOG(INFO) << "Layer " << id_ << " is invalid: "
      //             << "expected output dimension = [" << output_dimension_.str()
      //             << "], actual output dimension = [" << dims.str() << "]";
      //   } 
      // }
    }

// #ifdef GPU
//     else {
//       auto p_operation =
//           Factory<CudnnOperation>::CreateInstance(operation_name);
//       dims = p_operation->CalculateOutputDimension(*this, dims);
//     }
// #endif
  }

  // Perform dimension check after all operations have been applied
  if (dims.n() != output_dimension_.n() ||
      dims.c() != output_dimension_.c() ||
      dims.h() != output_dimension_.h() ||
      dims.w() != output_dimension_.w()) {
    LOG(ERROR) << "Layer " << id_ << " is invalid: "
               << "expected output dimension = [" << output_dimension_.str()
               << "], actual output dimension = [" << dims.str() << "]";
    return false;
  }

  LOG(INFO) << "----------------------------------------------------------------- " << id_;
  
  return true;
}



void Layer::id(int id) { id_ = id; }

int Layer::id() { return id_; }

void Layer::name(std::string name) { name_ = name; }

std::string Layer::name() { return name_; }

void Layer::type(const std::string &type) { type_ = type; }

const std::string &Layer::type() const { return type_; }

void Layer::predecessors(std::vector<int> predecessors) {
  predecessors_ = predecessors;
}

std::vector<int> &Layer::predecessors() { return predecessors_; }

int Layer::predecessors(int idx) { return predecessors_.at(idx); }

void Layer::successors(std::vector<int> successors) {
  successors_ = successors;
}

std::vector<int> &Layer::successors() { return successors_; }

int Layer::successors(int idx) { return successors_.at(idx); }

void Layer::inputs(const std::vector<Tensor> &activations) {
  int input_count = predecessors_.size();

  inputs_.clear();
  if (input_count == 0) {
    inputs_ = std::vector<std::shared_ptr<Tensor>>{
        std::make_shared<Tensor>(activations.at(0))};
  } else {
    inputs_.reserve(input_count);
    for (size_t index = 0; index < input_count; index++) {
      inputs_.push_back(
          std::make_shared<Tensor>(activations.at(predecessors_[index] + 1)));
    }
  }
}

std::vector<Dimension> &Layer::input_dimensions() { return input_dimensions_; }

void Layer::input_dimensions(std::vector<Dimension> dimensions) {
  input_dimensions_ = dimensions;
}

Dimension &Layer::input_dimensions(int idx) {
  return input_dimensions_.at(idx);
}

const Dimension &Layer::output_dimension() const { return output_dimension_; }

void Layer::output_dimension(const Dimension &dimension) {
  output_dimension_ = dimension;
}

void Layer::output_dimension(Dimension &&dimension) {
  output_dimension_ = std::move(dimension);
}

std::shared_ptr<Tensor> Layer::intermediate_activation() {
  return intermediate_activation_;
}

void Layer::intermediate_activation(std::shared_ptr<Tensor> act) {
  intermediate_activation_ = act;
}

void Layer::operation_types(const std::vector<std::string> &operation_types) {
  operation_types_ = operation_types;
}

const std::vector<std::string> &Layer::operation_types() const {
  return operation_types_;
}

const std::string &Layer::operation_types(int idx) const {
  return operation_types_.at(idx);
}

bool Layer::HasOperationTypes(const std::string &ops_type) {
  const std::vector<std::string> &operation_types = this->operation_types();
  const auto it = std::find(std::begin(operation_types),
                            std::end(operation_types), ops_type);

  return it != std::end(operation_types);
}

std::shared_ptr<Tensor> Layer::filter() const { return filter_; }

void Layer::filter(std::shared_ptr<Tensor> data) { filter_ = data; }

bool Layer::HasFilter() const {
  return filter_ != nullptr; 
}

std::shared_ptr<Tensor> Layer::bias() { return bias_; }

void Layer::bias(std::shared_ptr<Tensor> data) { bias_ = data; }

bool Layer::HasBias() { return bias_ != nullptr; }

std::shared_ptr<Tensor> Layer::scale() { return scale_; }

void Layer::scale(std::shared_ptr<Tensor> data) { scale_ = data; }

bool Layer::HasScale() { return scale_ != nullptr; }

std::shared_ptr<Tensor> Layer::mean() { return mean_; }

void Layer::mean(std::shared_ptr<Tensor> data) { mean_ = data; }

bool Layer::HasMean() { return mean_ != nullptr; }

std::shared_ptr<Tensor> Layer::variance() { return variance_; }

void Layer::variance(std::shared_ptr<Tensor> data) { variance_ = data; }

bool Layer::HasVariance() { return variance_ != nullptr; }

std::shared_ptr<Descriptor> Layer::convolution() { return convolution_; }

void Layer::convolution(std::shared_ptr<Descriptor> desc) {
  convolution_ = desc;
}

bool Layer::HasConvolutionDescriptor() { return convolution_ != nullptr; }

std::shared_ptr<Descriptor> Layer::sampling() { return sampling_; }

void Layer::sampling(std::shared_ptr<Descriptor> desc) { sampling_ = desc; }

bool Layer::HasSamplingDescriptor() { return sampling_ != nullptr; }

std::shared_ptr<Descriptor> Layer::ewadd() { return element_wise_add_; }

void Layer::ewadd(std::shared_ptr<Descriptor> desc) {
  element_wise_add_ = desc;
}

bool Layer::HasEwaddDescriptor() { return element_wise_add_ != nullptr; }

std::shared_ptr<Descriptor> Layer::ewmul() { return element_wise_mul_; }

void Layer::ewmul(std::shared_ptr<Descriptor> desc) {
  element_wise_mul_ = desc;
}

bool Layer::HasEwmulDescriptor() { return element_wise_mul_ != nullptr; }

std::string &Layer::activation_type() { return activation_type_; }

void Layer::activation_type(const std::string &value) {
  activation_type_ = value;
}

bool Layer::HasActivation() { return activation_type_ != ""; }

float Layer::epsilon() { return epsilon_; }

void Layer::epsilon(float value) { epsilon_ = value; }

float Layer::alpha() { return alpha_; }

void Layer::alpha(float value) { alpha_ = value; }

float Layer::gamma() { return gamma_; }

void Layer::gamma(float value) { gamma_ = value; }

int Layer::axis() { return axis_; }

void Layer::axis(int value) { axis_ = value; }

int Layer::stash_type() { return stash_type_; }

void Layer::stash_type(int value) { stash_type_ = value; }

float Layer::beta() { return beta_; }

void Layer::beta(float value) { beta_ = value; }

int64_t Layer::trans_A() { return trans_A_; }

void Layer::trans_A(int64_t value) { trans_A_ = value; }

int64_t Layer::trans_B() { return trans_B_; }

void Layer::trans_B(int64_t value) { trans_B_ = value; }

int64_t Layer::keepdims() { return keepdims_; }

void Layer::keepdims(int64_t value) { keepdims_ = value; }

int64_t Layer::noop_with_empty_axes() { return noop_with_empty_axes_; }

void Layer::noop_with_empty_axes(int64_t value) {
  noop_with_empty_axes_ = value;
}

int64_t Layer::select_last_index() { return select_last_index_; }

void Layer::select_last_index(int64_t value) { select_last_index_ = value; }

std::vector<int64_t> &Layer::axes() { return axes_; }

void Layer::axes(std::vector<int64_t> value) { axes_ = value; }

std::string Layer::mode() const { return mode_; }

void Layer::mode(const std::string value) { mode_ = value; }

float Layer::extrapolation_value() { return extrapolation_value_; }

void Layer::extrapolation_value(float value) { extrapolation_value_ = value; }

std::string Layer::coordinate_transformation_mode() const {
  return coordinate_transformation_mode_;
}

void Layer::coordinate_transformation_mode(const std::string value) {
  coordinate_transformation_mode_ = value;
}

std::string Layer::nearest_mode() const { return nearest_mode_; }

void Layer::nearest_mode(const std::string value) { nearest_mode_ = value; }

float Layer::cubic_coeff_a() { return cubic_coeff_a_; }

void Layer::cubic_coeff_a(float value) { cubic_coeff_a_ = value; }

int64_t Layer::exclude_outside() { return exclude_outside_; }

void Layer::exclude_outside(int64_t value) { exclude_outside_ = value; }

std::string Layer::auto_pad() const { return auto_pad_; }

void Layer::auto_pad(const std::string value) { auto_pad_ = value; }

int64_t Layer::group() { return group_; }

void Layer::group(int64_t value) { group_ = value; }

std::vector<int64_t> &Layer::dilations() { return dilations_; }

void Layer::dilations(std::vector<int64_t> value) { dilations_ = value; }

std::vector<int64_t> &Layer::kernel_shape() { return kernel_shape_; }

void Layer::kernel_shape(std::vector<int64_t> value) { kernel_shape_ = value; }

std::vector<int64_t> &Layer::output_padding() { return output_padding_; }

void Layer::output_padding(std::vector<int64_t> value) {
  output_padding_ = value;
}

std::vector<int64_t> &Layer::output_shape() { return output_shape_; }

void Layer::output_shape(std::vector<int64_t> value) { output_shape_ = value; }

std::vector<int64_t> &Layer::pads() { return pads_; }

void Layer::pads(std::vector<int64_t> value) { pads_ = value; }

std::vector<int64_t> &Layer::strides() { return strides_; }

void Layer::strides(std::vector<int64_t> value) { strides_ = value; }

std::vector<float> &Layer::input_thresholds() { return input_thresholds_; }

float Layer::input_thresholds(int idx) { return input_thresholds_.at(idx); }

void Layer::input_thresholds(std::vector<float> thresholds) {
  input_thresholds_ = thresholds;
}

float Layer::output_threshold() { return output_threshold_; }

void Layer::output_threshold(float threshold) { output_threshold_ = threshold; }

std::vector<float> &Layer::filter_thresholds() { return filter_thresholds_; }

void Layer::filter_thresholds(std::vector<float> thresolds) {
  filter_thresholds_ = thresolds;
}

x220::QuantConfig &Layer::x220_quant_config() { return *x220_quant_config_; }

void Layer::x220_quant_config(std::shared_ptr<x220::QuantConfig> quant_config) {
  x220_quant_config_ = std::move(quant_config);
}

x330::QuantConfig &Layer::x330_quant_config() { return *x330_quant_config_; }

void Layer::x330_quant_config(std::shared_ptr<x330::QuantConfig> quant_config) {
  x330_quant_config_ = std::move(quant_config);
}

std::vector<float> &Layer::negative_slope() { return neg_slopes_; }

float Layer::negative_slope(int idx) { return neg_slopes_.at(idx); }

void Layer::negative_slope(const std::vector<float> &neg_slopes) {
  neg_slopes_ = neg_slopes;
}
