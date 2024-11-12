#include "parser/spear_graph_parser.hpp"

#define BASE Parser
#define NAME spear_graph
#define CLASS SpearGraphParser
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

#include <any>
#include <fstream>
using std::fstream;

#include <limits>
#include <unordered_set>
using std::unordered_set;

#include <memory>
using std::make_shared;
using std::make_unique;
using std::shared_ptr;
using std::unique_ptr;

#include <string>
using std::ios;
using std::string;
using std::to_string;

#include <vector>
using std::vector;

#include "glog/logging.h"
#include "spear.proto.e8e8.pb.h"
using sapeon::simulator::SPGraph;
using sapeon::simulator::SPLayer;
using SPTensor = sapeon::simulator::SPLayer_SPTensor;
using SPConvolutionDesc = sapeon::simulator::SPLayer_SPConvolutionDesc;
using SPEWAddDesc = sapeon::simulator::SPLayer_SPEWAddDesc;
using SPEWMulDesc = sapeon::simulator::SPLayer_SPEWMulDesc;
using SPSamplingDesc = sapeon::simulator::SPLayer_SPSamplingDesc;

#include "datatype.hpp"
using dty::DataType;
#include "enums/activation_names.hpp"
#include "enums/error.hpp"
#include "enums/operation_names.hpp"
#include "factory.hpp"
#include "network/descriptor.hpp"
#include "network/dimension.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "tl/expected.hpp"
using tl::expected;
using tl::make_unexpected;
using tl::unexpected;

namespace parser {

class AttributeNames {
 public:
  static constexpr const char *const kLeakySlope = "leaky_slope";
  static constexpr const char *const kNegativeSlope = "neg_slope";
};

static bool kRegistered =
    Factory<BASE>::RegisterCreateFunction(GET_STR(NAME), CLASS::CreateParser);

unique_ptr<BASE> SCOPE::CreateParser() { return make_unique<CLASS>(); }

expected<void, SimulatorError> SCOPE::ReadGraphBinary(
    const string &binary_path) {
  DLOG(INFO) << "Start to Read Spear Graph Binary\n";
  fstream fs(binary_path, ios::in | ios::binary);

  spear_graph_.ParseFromIstream(&fs);
  return {};
}

expected<void, SimulatorError> SCOPE::ParseGraphBinary(
    unique_ptr<Network> &network) {
  const int num_layers = spear_graph_.layer().size();

  network->num_layers(num_layers);
  network->layers(vector<Layer>(num_layers));
  network->num_operations(vector<int>(num_layers));
  network->graph_type("spear_graph");

  auto &layers = network->layers();
  auto &num_operations = network->num_operations();
  for (int idx_layer = 0; idx_layer < num_layers; ++idx_layer) {
    auto e = ParseSPLayer(spear_graph_.layer(idx_layer));
    if (!e) return unexpected<SimulatorError>(e.error());
    layers.at(idx_layer) = e.value();
    num_operations.at(idx_layer) = spear_graph_.layer(idx_layer).type().size();
  }
  return {};
}

expected<Layer, SimulatorError> SCOPE::ParseSPLayer(
    const SPLayer &spear_layer) {
  auto layer = Layer();
  const int group = spear_layer.convdesc().groups();

  //! FIXME: Currently, calibration table format requires layer name, but this
  //! may be changed in the future.
  if (!spear_layer.has_name()) {
    LOG(ERROR) << "SPLayer Name is missing"
               << "\n";
  }
  layer.name(spear_layer.name());
  layer.id(spear_layer.id());

  auto num_predecessors = spear_layer.preds().size();
  layer.predecessors(vector<int>(num_predecessors));
  auto &predecessors = layer.predecessors();
  for (int i = 0; i < num_predecessors; ++i) {
    predecessors.at(i) = spear_layer.preds(i);
  }

  layer.successors(vector<int>(spear_layer.succs().size()));
  auto &successors = layer.successors();
  for (int i = 0; i < spear_layer.succs().size(); ++i) {
    successors.at(i) = spear_layer.succs(i);
  }

  int num_inputs = num_predecessors;
  if (num_predecessors == 0) {
    // currently first layer has only one input
    num_inputs = 1;
  }
  layer.input_dimensions(vector<Dimension>(num_inputs));
  auto &dimensions = layer.input_dimensions();
  if (num_predecessors != 0) {
    for (int i = 0; i < num_predecessors; ++i) {
      const SPLayer &preds_layer = spear_graph_.layer(spear_layer.preds(i));
      dimensions.at(i) =
          Dimension(preds_layer.output().dims(3), preds_layer.output().dims(2),
                    preds_layer.output().dims(1), preds_layer.output().dims(0));
    }
  } else {
    dimensions.at(0) =
        Dimension(spear_layer.input(0).dims(3), spear_layer.input(0).dims(2),
                  spear_layer.input(0).dims(1), spear_layer.input(0).dims(0));
  }

  layer.output_dimension(
      Dimension(spear_layer.output().dims(3), spear_layer.output().dims(2),
                spear_layer.output().dims(1), spear_layer.output().dims(0)));

  if (spear_layer.type().size() > 0) {
    layer.type(spear_layer.type(0));
  }

  auto operation_types = vector<string>(spear_layer.type().size());
  for (int i = 0; i < spear_layer.type().size(); ++i) {
    operation_types.at(i) =
        GetOperationName(spear_layer.type(i), "spear_graph");
  }
  layer.operation_types(operation_types);

  if (spear_layer.has_filter() && spear_layer.filter().fval().size()) {
    auto e = ParseSPTensor(spear_layer.filter(), group);
    if (!e) return unexpected<SimulatorError>(e.error());
    layer.filter(e.value());
  } else {
    layer.filter(nullptr);
  }

  if (spear_layer.has_bias() && spear_layer.bias().fval().size()) {
    auto e = ParseSPTensor(spear_layer.bias(), group);
    if (!e) return unexpected<SimulatorError>(e.error());
    layer.bias(e.value());
  } else {
    layer.bias(nullptr);
  }

  if (spear_layer.has_scale() && spear_layer.scale().fval().size()) {
    auto e = ParseSPTensor(spear_layer.scale(), group);
    if (!e) return unexpected<SimulatorError>(e.error());
    layer.scale(e.value());
  } else {
    layer.scale(nullptr);
  }

  if (spear_layer.has_mean() && spear_layer.mean().fval().size()) {
    auto e = ParseSPTensor(spear_layer.mean(), group);
    if (!e) return unexpected<SimulatorError>(e.error());
    layer.mean(e.value());
  } else {
    layer.mean(nullptr);
  }

  if (spear_layer.has_variance() && spear_layer.variance().fval().size()) {
    auto e = ParseSPTensor(spear_layer.variance(), group);
    if (!e) return unexpected<SimulatorError>(e.error());
    layer.variance(e.value());
  } else {
    layer.variance(nullptr);
  }

  if (spear_layer.has_convdesc()) {
    auto e = ParseSPConvolutionDesc(spear_layer.convdesc(), spear_layer);
    if (!e) return unexpected<SimulatorError>(e.error());
    layer.convolution(e.value());
  }

  if (spear_layer.has_samplingdesc()) {
    auto e = ParseSPSamplingDesc(spear_layer.samplingdesc(), spear_layer);
    if (!e) return unexpected<SimulatorError>(e.error());
    layer.sampling(e.value());
  }

  if (spear_layer.has_ewadddesc()) {
    auto e = ParseSPEWAddDesc(spear_layer.ewadddesc(), spear_layer);
    if (!e) return unexpected<SimulatorError>(e.error());
    layer.ewadd(e.value());
  }

  if (spear_layer.has_ewmuldesc()) {
    auto e = ParseSPEWMulDesc(spear_layer.ewmuldesc(), spear_layer);
    if (!e) return unexpected<SimulatorError>(e.error());
    layer.ewmul(e.value());
  }

  if (spear_layer.has_activation()) {
    layer.activation_type(
        GetActivationName(spear_layer.activation(), "spear_graph"));

    if (spear_layer.activation() == SPLayer::SP_ACTIVATION_LEAKY_RELU ||
        spear_layer.activation() == SPLayer::SP_ACTIVATION_PRELU) {
      auto res = ParseSPAttribute(layer.activation_type(), spear_layer);
      if (res) {
        layer.negative_slope(
            std::move(std::any_cast<std::vector<float>>(res.value())));
        if (layer.negative_slope().size() > 1) {
          //! FIXME: should change the CWPrelu literal to proto generated
          //! constant.
          layer.activation_type("CWPrelu");
        } else {
          layer.activation_type(
              GetActivationName(SPLayer::SP_ACTIVATION_PRELU, "spear_graph"));
        }
      } else {
        return make_unexpected(res.error());
      }
    }
  }

  if (spear_layer.has_epsilon()) {
    layer.epsilon(spear_layer.epsilon());
  }

  layer.input_thresholds(vector<float>(spear_layer.input_threshold().size()));
  auto &input_thresholds = layer.input_thresholds();
  for (int i = 0; i < spear_layer.input_threshold().size(); ++i) {
    input_thresholds.at(i) = spear_layer.input_threshold(i);
  }

  if (spear_layer.has_output_threshold()) {
    layer.output_threshold(spear_layer.output_threshold());
  }

  layer.filter_thresholds(vector<float>(spear_layer.filter_threshold().size()));
  auto &filter_thresholds = layer.filter_thresholds();
  for (int i = 0; i < spear_layer.filter_threshold().size(); ++i) {
    filter_thresholds.at(i) = spear_layer.filter_threshold(i);
  }

  return layer;
}

expected<shared_ptr<Tensor>, SimulatorError> SCOPE::ParseSPTensor(
    const SPTensor &spear_tensor, const int group) {
  DataType dtype = static_cast<DataType>(spear_tensor.dtype());
  shared_ptr<Tensor> tensor;

  if (spear_tensor.dims().size() == 1) {
    tensor = std::make_shared<Tensor>(spear_tensor.dims(0), dtype);
  } else if (spear_tensor.dims().size() == 4) {
    int channel = spear_tensor.dims(2);
    if (group != 1) {
      channel /= group;
    }
    tensor = std::make_shared<Tensor>(spear_tensor.dims(3), channel,
                                      spear_tensor.dims(1),
                                      spear_tensor.dims(0), dtype);
  } else {
    auto dims = vector<size_t>(spear_tensor.dims().size());
    for (int i = 0; i < spear_tensor.dims().size(); ++i) {
      dims.at(i) = static_cast<size_t>(spear_tensor.dims(i));
    }
    tensor = std::make_shared<Tensor>(std::move(dims), dtype);
  }

  void *data = tensor->data();

  switch (dtype) {
    case dty::DataType::FP32:
      if (spear_tensor.fval_size() == tensor->dimension().size()) {
        memcpy(data, spear_tensor.fval().data(), tensor->size());
      }
      break;
    case dty::DataType::INT8:
    case dty::DataType::UINT8:
      if (spear_tensor.bval_size() == tensor->dimension().size()) {
        memcpy(data, spear_tensor.bval().data(), tensor->size());
      }
      break;
    case dty::DataType::INT16:
    case dty::DataType::FP64:
    case dty::DataType::FP16:
    default:
      const string msg =
          "`" + dty::NameOf(dtype) + "` is not supported for tensor";
      LOG(ERROR) << msg;
      return unexpected<SimulatorError>(SimulatorError::kModelParsingError);
  }

  return tensor;
}

expected<shared_ptr<Descriptor>, SimulatorError> SCOPE::ParseSPConvolutionDesc(
    const SPConvolutionDesc &convdesc, const SPLayer &layer) {
  auto desc = make_shared<Descriptor>();

  if (convdesc.padding_size() != 4) {
    const string msg =
        "Assertion failed: `SPConvolutionDesc::padding_size() == 4`, "
        "found: " +
        to_string(convdesc.padding_size());
    LOG(ERROR) << msg;
    return unexpected<SimulatorError>(SimulatorError::kModelParsingError);
  }
  desc->padding_height_top(convdesc.padding(0));
  desc->padding_height_bottom(convdesc.padding(1));
  desc->padding_width_left(convdesc.padding(2));
  desc->padding_width_right(convdesc.padding(3));

  if (convdesc.stride_size() < 2) {
    const string msg =
        "Assertion failed: `SPConvolutionDesc::stride_size() >= 2`, found: " +
        to_string(convdesc.stride_size());
    LOG(ERROR) << msg;
    return unexpected<SimulatorError>(SimulatorError::kModelParsingError);
  }
  desc->stride_width(convdesc.stride(0));
  desc->stride_height(convdesc.stride(1));

  if (convdesc.dilation_size() < 2) {
    const string msg =
        "Assertion failed: `SPConvolutionDesc::dilation_size() >= 2`, "
        "found: " +
        to_string(convdesc.dilation_size());
    LOG(ERROR) << msg;
    return unexpected<SimulatorError>(SimulatorError::kModelParsingError);
  }
  desc->dilation_width(convdesc.dilation(0));
  desc->dilation_height(convdesc.dilation(1));

  if (!convdesc.has_groups()) {
    const string msg =
        "Assertion failed: `SPConvolutionDesc::has_groups() == true`, found: "
        "false";
    LOG(ERROR) << msg;
    return unexpected<SimulatorError>(SimulatorError::kModelParsingError);
  }
  desc->groups(convdesc.groups());

  return desc;
}

expected<std::shared_ptr<Descriptor>, SimulatorError> SCOPE::ParseSPEWAddDesc(
    const SPEWAddDesc &ewadddesc, const SPLayer &layer) {
  auto desc = make_shared<Descriptor>();
  if (ewadddesc.scale_size() != layer.preds_size()) {
    const string msg =
        "Assertion failed: `SPEWAddDesc::scale_size() == "
        "SPLayer::preds_size()`, found: scale_size() = " +
        to_string(ewadddesc.scale_size()) +
        ", preds_size() = " + to_string(layer.preds_size());
    LOG(ERROR) << msg;
    return unexpected<SimulatorError>(SimulatorError::kModelParsingError);
  }
  // TO-DO: need to check if desc->scale should be vector
  desc->scale(ewadddesc.scale(0));
  return desc;
}

expected<std::shared_ptr<Descriptor>, SimulatorError> SCOPE::ParseSPEWMulDesc(
    const SPEWMulDesc &ewmuldesc, const SPLayer &layer) {
  auto desc = make_shared<Descriptor>();
  if (ewmuldesc.scale_size() != layer.preds_size()) {
    const string msg =
        "Assertion failed: `SPEWMulDesc::scale_size() == "
        "SPLayer::preds_size()`, found: scale_size() = " +
        to_string(ewmuldesc.scale_size()) +
        ", preds_size() = " + to_string(layer.preds_size());
    LOG(ERROR) << msg;
    return unexpected<SimulatorError>(SimulatorError::kModelParsingError);
  }
  // TO-DO: need to check if desc->scale should be vector
  desc->scale(ewmuldesc.scale(0));
  return desc;
}

expected<std::shared_ptr<Descriptor>, SimulatorError>
SCOPE::ParseSPSamplingDesc(const SPSamplingDesc &samplingdesc,
                           const SPLayer &layer) {
  auto desc = make_shared<Descriptor>();

  // FIXME: pixelshuffle do not have window
  if (samplingdesc.window_size() >= 2) {
    desc->window_width(samplingdesc.window(0));
    desc->window_height(samplingdesc.window(1));
  }

  if (samplingdesc.stride_size() < 2) {
    const string msg =
        "Assertion failed: `SPSamplingDesc::stride_size() >= 2`, found: " +
        to_string(samplingdesc.stride_size());
    LOG(ERROR) << msg;
    return unexpected<SimulatorError>(SimulatorError::kModelParsingError);
  }
  desc->stride_width(samplingdesc.stride(0));
  desc->stride_height(samplingdesc.stride(1));

  if (samplingdesc.padding_size() < 4) {
    const string msg =
        "Assertion failed: `SPSamplingDesc::padding_size() >= 4`, found: " +
        to_string(samplingdesc.padding_size());
    LOG(ERROR) << msg;
    return unexpected<SimulatorError>(SimulatorError::kModelParsingError);
  }
  desc->padding_height_top(samplingdesc.padding(0));
  desc->padding_height_bottom(samplingdesc.padding(1));
  desc->padding_width_left(samplingdesc.padding(2));
  desc->padding_width_right(samplingdesc.padding(3));

  return desc;
}

expected<void, SimulatorError> SCOPE::DumpGraphBinary(
    const string &binary_path) {
  fstream new_spear_graph(binary_path.c_str(), ios::out | ios::binary);
  if (!new_spear_graph) {
    const string msg = "Failed to open file: `" + binary_path + "`";
    LOG(ERROR) << msg;
    return unexpected<SimulatorError>(SimulatorError::kFileWriteError);
  }

  bool result = spear_graph_.SerializeToOstream(&new_spear_graph);
  if (!result) {
    const string msg = "Failed to dump graph: `" + binary_path + "`";
    LOG(ERROR) << msg;
    return unexpected<SimulatorError>(SimulatorError::kFileWriteError);
  }
  return {};
}

expected<void, SimulatorError> SCOPE::UpdateGraphThresholds(
    unique_ptr<Network> &network) {
  const size_t num_layers = network->num_layers();

  const auto &input_thresholds = network->layers(0).input_thresholds();
  for (auto idx = 0; idx < input_thresholds.size(); idx++) {
    spear_graph_.mutable_layer(0)->set_input_threshold(
        idx, input_thresholds.at(idx));
  }

  for (size_t i = 0; i < num_layers; ++i) {
    const float output_threshold = network->layers(i).output_threshold();
    spear_graph_.mutable_layer(i)->set_output_threshold(output_threshold);

    const auto &filter_thresholds = network->layers(i).filter_thresholds();
    for (auto idx = 0; idx < filter_thresholds.size(); idx++) {
      spear_graph_.mutable_layer(i)->set_filter_threshold(
          idx, filter_thresholds.at(idx));
    }
  }
  return {};
}

expected<std::any, SimulatorError> SCOPE::ParseSPAttribute(
    const std::string &attr_name, const sapeon::simulator::SPLayer &layer) {
  const auto &attrs = layer.attributes();
  auto find_attr = [&attrs](const std::string &name) {
    for (auto iter = attrs.cbegin(); iter != attrs.cend(); iter++) {
      if (iter->has_name() && name == iter->name()) {
        return iter;
      }
    }
    return attrs.cend();
  };

  const auto &iter = find_attr(attr_name);
  if (attr_name == AttributeNames::kLeakySlope ||
      AttributeNames::kNegativeSlope) {
    std::vector<float> ret;

    if (iter == attrs.cend()) {
      ret.push_back(1.0);
      return ret;
    }

    if (iter->has_f()) {
      ret.push_back(iter->f());
    } else if (iter->floats_size() > 0) {
      for (auto elem : iter->floats()) {
        ret.push_back(elem);
      }
    } else {
      LOG(ERROR) << attr_name + " has invalid values\n";
      return make_unexpected(SimulatorError::kModelParsingError);
    }
    return ret;
  }

  LOG(ERROR) << "Unsupported attribute name: " + attr_name + "\n";
  return make_unexpected(SimulatorError::kModelParsingError);
}

}  // namespace parser
