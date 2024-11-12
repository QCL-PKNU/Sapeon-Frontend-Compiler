#include "parser/aix_graph_parser.hpp"

#define BASE parser::Parser
#define NAME aix_graph
#define CLASS parser::AIXGraphParser
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

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

#include <vector>
using std::vector;

#include "aixh.pb.h"
#include "glog/logging.h"
using aixh::AIXGraph;
using aixh::AIXLayer;
using AIXTensor = aixh::AIXLayer_AIXTensor;
using AIXConvolutionDesc = aixh::AIXLayer_AIXConvolutionDesc;
using AIXEWAddDesc = aixh::AIXLayer_AIXEWAddDesc;
using AIXSamplingDesc = aixh::AIXLayer_AIXSamplingDesc;

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
using tl::unexpected;

static bool kRegistered =
    Factory<BASE>::RegisterCreateFunction(GET_STR(NAME), CLASS::CreateParser);

unique_ptr<BASE> SCOPE::CreateParser() { return make_unique<CLASS>(); }

expected<void, SimulatorError> SCOPE::ReadGraphBinary(
    const string &binary_path) {
  LOG(INFO) << "Start to Read AIX Graph Binary\n";
  fstream fs(binary_path, ios::in | ios::binary);
  if (!fs) {
    const string msg = "Failed to open file: `" + binary_path + "`";
    LOG(ERROR) << msg;
    return unexpected<SimulatorError>(SimulatorError::kFileReadError);
  }

  bool result = aix_graph_.ParseFromIstream(&fs);
  if (!result) {
    const string msg =
        "Failed to parse graph: `" + binary_path + "` is not a valid AIX Graph";
    LOG(ERROR) << msg;
    return unexpected<SimulatorError>(SimulatorError::kModelParsingError);
  }

  LOG(INFO) << "Fininhed to Read AIX Graph Binary\n";
  return {};
}

expected<void, SimulatorError> SCOPE::ParseGraphBinary(
    unique_ptr<Network> &network) {
  const int num_layers = aix_graph_.layer().size();

  LOG(INFO) << "Starting to parse AIX Graph Binary with " << num_layers << " layers";

  // Ensure that the number of layers is valid
  if (num_layers <= 0) {
    LOG(ERROR) << "Invalid number of layers: " << num_layers;
    return unexpected<SimulatorError>(SimulatorError::kInvalidModel);
  }

  network->num_layers(num_layers);

  LOG(INFO) << "Setting number of layers to " << num_layers;

  network->layers(vector<Layer>(num_layers));
  network->num_operations(vector<int>(num_layers));

  LOG(INFO) << "Network initialized with layers and operations.";

  network->graph_type("aix_graph");

  auto &layers = network->layers();
  auto &num_operations = network->num_operations();

  for (int idx_layer = 0; idx_layer < num_layers; ++idx_layer) {
    // Ensure that we are within bounds of the layers vector
    if (idx_layer >= layers.size()) {
      LOG(ERROR) << "Layer index " << idx_layer << " out of bounds!";
      return unexpected<SimulatorError>(SimulatorError::kInvalidModel);
    }

    layers.at(idx_layer) = ParseAIXLayer(aix_graph_.layer(idx_layer));

    // Ensure the layer has valid types
    if (aix_graph_.layer(idx_layer).type().size() > 0) {
      num_operations.at(idx_layer) = aix_graph_.layer(idx_layer).type().size();
    } else {
      LOG(ERROR) << "Layer " << idx_layer << " has no valid types!";
      return unexpected<SimulatorError>(SimulatorError::kInvalidModel);
    }

    LOG(INFO) << "Finished parsing layer " << idx_layer << " with "
              << num_operations.at(idx_layer) << " operations.";
  }

  LOG(INFO) << "Finished parsing AIX Graph Binary successfully";
  return {};
}


Layer SCOPE::ParseAIXLayer(const AIXLayer &aix_layer) {
  auto layer = Layer();
  const int group = aix_layer.convdesc().groups();

  auto num_predecessors = aix_layer.preds().size();
  layer.predecessors(vector<int>(num_predecessors));
  auto &predecessors = layer.predecessors();
  for (int i = 0; i < num_predecessors; ++i) {
    predecessors.at(i) = aix_layer.preds(i);
  }

  layer.successors(vector<int>(aix_layer.succs().size()));
  auto &successors = layer.successors();
  for (int i = 0; i < aix_layer.succs().size(); ++i) {
    successors.at(i) = aix_layer.succs(i);
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
      const AIXLayer &preds_layer = aix_graph_.layer(aix_layer.preds(i));
      dimensions.at(i) =
          Dimension(preds_layer.output().dims(3), preds_layer.output().dims(2),
                    preds_layer.output().dims(1), preds_layer.output().dims(0));
    }
  } else {
    dimensions.at(0) =
        Dimension(aix_layer.input().dims(3), aix_layer.input().dims(2),
                  aix_layer.input().dims(1), aix_layer.input().dims(0));
  }

  layer.output_dimension(
      Dimension(aix_layer.output().dims(3), aix_layer.output().dims(2),
                aix_layer.output().dims(1), aix_layer.output().dims(0)));

  auto operation_types = vector<string>(aix_layer.type().size());
  for (int i = 0; i < aix_layer.type().size(); ++i) {
    auto operation_key = static_cast<int>(aix_layer.type(i));
    operation_types.at(i) =
        GetOperationName(std::to_string(operation_key), "aix_graph");
  }
  layer.operation_types(operation_types);

  if (aix_layer.has_filter() && aix_layer.filter().fval().size()) {
    layer.filter(ParseAIXTensor(aix_layer.filter(), group));
  } else {
    layer.filter(nullptr);
  }

  if (aix_layer.has_bias() && aix_layer.bias().fval().size()) {
    layer.bias(ParseAIXTensor(aix_layer.bias(), group));
  } else {
    layer.bias(nullptr);
  }

  if (aix_layer.has_scale() && aix_layer.scale().fval().size()) {
    layer.scale(ParseAIXTensor(aix_layer.scale(), group));
  } else {
    layer.scale(nullptr);
  }

  if (aix_layer.has_mean() && aix_layer.mean().fval().size()) {
    layer.mean(ParseAIXTensor(aix_layer.mean(), group));
  } else {
    layer.mean(nullptr);
  }

  if (aix_layer.has_variance() && aix_layer.variance().fval().size()) {
    layer.variance(ParseAIXTensor(aix_layer.variance(), group));
  } else {
    layer.variance(nullptr);
  }

  if (aix_layer.has_convdesc()) {
    layer.convolution(ParseAIXConvolutionDesc(aix_layer.convdesc()));
  }

  if (aix_layer.has_samplingdesc()) {
    layer.sampling(ParseAIXSamplingDesc(aix_layer.samplingdesc()));
  }

  if (aix_layer.has_ewadddesc()) {
    layer.ewadd(ParseAIXEWAddDesc(aix_layer.ewadddesc()));
  }

  layer.activation_type(GetActivationName(aix_layer.activation(), "aix_graph"));

  if (aix_layer.has_epsilon()) {
    layer.epsilon(aix_layer.epsilon());
  }

  if (aix_layer.has_alpha()) {
    layer.alpha(aix_layer.alpha());
  } else {
    layer.alpha(std::numeric_limits<float>::quiet_NaN());
  }

  if (aix_layer.has_gamma()) {
    layer.gamma(aix_layer.gamma());
  } else {
    layer.gamma(std::numeric_limits<float>::quiet_NaN());
  }

  if (aix_layer.has_axis()) {
    layer.axis(aix_layer.axis());
  } else {
    layer.axis(std::numeric_limits<int>::lowest());
  }

  if (aix_layer.has_stash_type()) {
    layer.stash_type(aix_layer.stash_type());
  } else {
    layer.stash_type(std::numeric_limits<int>::lowest());
  }

  if (aix_layer.has_beta()) {
    layer.beta(aix_layer.beta());
  } else {
    layer.beta(std::numeric_limits<float>::quiet_NaN());
  }

  if (aix_layer.has_transa()) {
    layer.trans_A(aix_layer.transa());
  } else {
    layer.trans_A(std::numeric_limits<int64_t>::lowest());
  }

  if (aix_layer.has_transb()) {
    layer.trans_B(aix_layer.transb());
  } else {
    layer.trans_B(std::numeric_limits<int64_t>::lowest());
  }

  if (aix_layer.has_keepdims()) {
    layer.keepdims(aix_layer.keepdims());
  } else {
    layer.keepdims(std::numeric_limits<int64_t>::lowest());
  }

  if (aix_layer.has_noop_with_empty_axes()) {
    layer.noop_with_empty_axes(aix_layer.noop_with_empty_axes());
  } else {
    layer.noop_with_empty_axes(std::numeric_limits<int64_t>::lowest());
  }

  if (aix_layer.has_select_last_index()) {
    layer.select_last_index(aix_layer.select_last_index());
  } else {
    layer.select_last_index(std::numeric_limits<int64_t>::lowest());
  }

  layer.axes(vector<int64_t>(aix_layer.axes().size()));
  auto &axes = layer.axes();
  for (int i = 0; i < aix_layer.axes().size(); ++i) {
    axes.at(i) = aix_layer.axes(i);
  }

  if (aix_layer.has_mode()) {
    layer.mode(aix_layer.mode());
  }

  if (aix_layer.has_extrapolation_value()) {
    layer.extrapolation_value(aix_layer.extrapolation_value());
  } else {
    layer.extrapolation_value(std::numeric_limits<float>::quiet_NaN());
  }

  if (aix_layer.has_coordinate_transformation_mode()) {
    layer.coordinate_transformation_mode(
        aix_layer.coordinate_transformation_mode());
  }

  if (aix_layer.has_nearest_mode()) {
    layer.nearest_mode(aix_layer.nearest_mode());
  }

  if (aix_layer.has_cubic_coeff_a()) {
    layer.cubic_coeff_a(aix_layer.cubic_coeff_a());
  } else {
    layer.cubic_coeff_a(std::numeric_limits<float>::quiet_NaN());
  }

  if (aix_layer.has_exclude_outside()) {
    layer.exclude_outside(aix_layer.exclude_outside());
  } else {
    layer.exclude_outside(std::numeric_limits<int64_t>::lowest());
  }

  if (aix_layer.has_auto_pad()) {
    layer.auto_pad(aix_layer.auto_pad());
  }

  if (aix_layer.has_group()) {
    layer.group(aix_layer.group());
  }

  layer.dilations(vector<int64_t>(aix_layer.dilations().size()));
  auto &dilations = layer.dilations();
  for (int i = 0; i < aix_layer.dilations().size(); ++i) {
    dilations.at(i) = aix_layer.dilations(i);
  }

  layer.kernel_shape(vector<int64_t>(aix_layer.kernel_shape().size()));
  auto &kernel_shape = layer.kernel_shape();
  for (int i = 0; i < aix_layer.kernel_shape().size(); ++i) {
    kernel_shape.at(i) = aix_layer.kernel_shape(i);
  }

  layer.output_padding(vector<int64_t>(aix_layer.output_padding().size()));
  auto &output_padding = layer.output_padding();
  for (int i = 0; i < aix_layer.output_padding().size(); ++i) {
    output_padding.at(i) = aix_layer.output_padding(i);
  }

  layer.output_shape(vector<int64_t>(aix_layer.output_shape().size()));
  auto &output_shape = layer.output_shape();
  for (int i = 0; i < aix_layer.output_shape().size(); ++i) {
    output_shape.at(i) = aix_layer.output_shape(i);
  }

  layer.pads(vector<int64_t>(aix_layer.pads().size()));
  auto &pads = layer.pads();
  for (int i = 0; i < aix_layer.pads().size(); ++i) {
    pads.at(i) = aix_layer.pads(i);
  }

  layer.strides(vector<int64_t>(aix_layer.strides().size()));
  auto &strides = layer.strides();
  for (int i = 0; i < aix_layer.strides().size(); ++i) {
    strides.at(i) = aix_layer.strides(i);
  }

  layer.input_thresholds(vector<float>(1));
  auto &input_thresholds = layer.input_thresholds();
  input_thresholds.at(0) = aix_layer.input_threshold();

  if (aix_layer.has_output_threshold()) {
    layer.output_threshold(aix_layer.output_threshold());
  }

  layer.filter_thresholds(vector<float>(aix_layer.filter_threshold().size()));
  auto &filter_thresholds = layer.filter_thresholds();
  for (int i = 0; i < aix_layer.filter_threshold().size(); ++i) {
    filter_thresholds.at(i) = aix_layer.filter_threshold(i);
  }

  return layer;
}


// Layer SCOPE::ParseAIXLayer(const AIXLayer &aix_layer) {
//   auto layer = Layer();
//   const int group = aix_layer.convdesc().groups();

//   // Check and parse predecessors
//   auto num_predecessors = aix_layer.preds().size();
//   layer.predecessors(vector<int>(num_predecessors));
//   auto &predecessors = layer.predecessors();
//   for (int i = 0; i < num_predecessors; ++i) {
//     LOG(INFO) << "Parsing predecessor " << i << " for layer.";
//     if (i < aix_layer.preds().size()) {
//       predecessors.at(i) = aix_layer.preds(i);
//     } else {
//       LOG(ERROR) << "Predecessor index out of bounds.";
//       throw std::out_of_range("Predecessor index out of bounds.");
//     }
//   }

//   // Check and parse successors
//   layer.successors(vector<int>(aix_layer.succs().size()));
//   auto &successors = layer.successors();
//   for (int i = 0; i < aix_layer.succs().size(); ++i) {
//     LOG(INFO) << "Parsing successor " << i << " for layer.";
//     if (i < aix_layer.succs().size()) {
//       successors.at(i) = aix_layer.succs(i);
//     } else {
//       LOG(ERROR) << "Successor index out of bounds.";
//       throw std::out_of_range("Successor index out of bounds.");
//     }
//   }

//   LOG(INFO) << "------------------------";


//   int num_inputs = num_predecessors;
//   if (num_predecessors == 0) {
//     // currently first layer has only one input
//     num_inputs = 1;
//   }
//   layer.input_dimensions(vector<Dimension>(num_inputs));
//   auto &dimensions = layer.input_dimensions();
//   if (num_predecessors != 0) {
//     for (int i = 0; i < num_predecessors; ++i) {
//       const AIXLayer &preds_layer = aix_graph_.layer(aix_layer.preds(i));
//       dimensions.at(i) =
//           Dimension(preds_layer.output().dims(3), preds_layer.output().dims(2),
//                     preds_layer.output().dims(1), preds_layer.output().dims(0));
//     }
//   } else {
//     dimensions.at(0) =
//         Dimension(aix_layer.input().dims(3), aix_layer.input().dims(2),
//                   aix_layer.input().dims(1), aix_layer.input().dims(0));
//   }

//   // Check output dimension bounds
//   if (aix_layer.output().dims_size() >= 4) {
//     layer.output_dimension(
//         Dimension(aix_layer.output().dims(3), aix_layer.output().dims(2),
//                   aix_layer.output().dims(1), aix_layer.output().dims(0)));
//   } else {
//     LOG(ERROR) << "Layer output dimensions are not sufficient!";
//     throw std::out_of_range("Layer output dimensions are not sufficient!");
//   }

//   // // Check and parse layer types (Place the type parsing here)
//   // if (aix_layer.type().size() > 0) {
//   //     auto operation_types = vector<string>(aix_layer.type().size());
//   //     for (int i = 0; i < aix_layer.type().size(); ++i) {
//   //         if (i < aix_layer.type().size()) {  // Add bounds checking
//   //             LOG(INFO) << "Parsing layer type " << i;
//   //             auto operation_key = static_cast<int>(aix_layer.type(i));
//   //             operation_types.at(i) = GetOperationName(std::to_string(operation_key), "aix_graph");
//   //         } else {
//   //             LOG(ERROR) << "Layer type index " << i << " out of bounds!";
//   //             throw std::out_of_range("Layer type index out of bounds.");
//   //         }
//   //     }
//   //     layer.operation_types(operation_types);
//   // } else {
//   //     LOG(ERROR) << "Layer has no types!";
//   //     throw std::out_of_range("Layer has no types.");
//   // }

//   // LOG(INFO) << "Finished parsing layer types.";


//   // Check and parse layer types
//   if (aix_layer.type().size() > 0) {
//     auto operation_types = vector<string>(aix_layer.type().size());
//     for (int i = 0; i < aix_layer.type().size(); ++i) {
//       LOG(INFO) << "Parsing layer type " << i;
//       auto operation_key = static_cast<int>(aix_layer.type(i));
//       operation_types.at(i) =
//           GetOperationName(std::to_string(operation_key), "aix_graph");
//     }
//     layer.operation_types(operation_types);
//   } else {
//     LOG(ERROR) << "Layer has no types!";
//     throw std::out_of_range("Layer has no types.");
//   }

//   LOG(INFO) << "Finished parsing layer types.";

//   // Filter, bias, scale, mean, and variance
//   if (aix_layer.has_filter() && aix_layer.filter().fval().size()) {
//     layer.filter(ParseAIXTensor(aix_layer.filter(), group));
//   } else {
//     layer.filter(nullptr);
//   }

//   if (aix_layer.has_bias() && aix_layer.bias().fval().size()) {
//     layer.bias(ParseAIXTensor(aix_layer.bias(), group));
//   } else {
//     layer.bias(nullptr);
//   }


//   LOG(INFO) << "-------- / Finished parsing layer types.";

//   if (aix_layer.has_scale() && aix_layer.scale().fval().size()) {
//     layer.scale(ParseAIXTensor(aix_layer.scale(), group));
//   } else {
//     layer.scale(nullptr);
//   }



//   LOG(INFO) << "-------- // Finished parsing layer types.";

//   if (aix_layer.has_mean() && aix_layer.mean().fval().size()) {
//     layer.mean(ParseAIXTensor(aix_layer.mean(), group));
//   } else {
//     layer.mean(nullptr);
//   }


//   LOG(INFO) << "-------- /// Finished parsing layer types.";

//   if (aix_layer.has_variance() && aix_layer.variance().fval().size()) {
//     layer.variance(ParseAIXTensor(aix_layer.variance(), group));
//   } else {
//     layer.variance(nullptr);
//   }

//   if (aix_layer.has_convdesc()) {
//     layer.convolution(ParseAIXConvolutionDesc(aix_layer.convdesc()));
//   }


//   if (aix_layer.has_samplingdesc()) {
//     layer.sampling(ParseAIXSamplingDesc(aix_layer.samplingdesc()));
//   }


//   if (aix_layer.has_ewadddesc()) {
//     layer.ewadd(ParseAIXEWAddDesc(aix_layer.ewadddesc()));
//   }

//   layer.activation_type(GetActivationName(aix_layer.activation(), "aix_graph"));


//   if (aix_layer.has_epsilon()) {
//     layer.epsilon(aix_layer.epsilon());
//   }

//   if (aix_layer.has_alpha()) {
//     layer.alpha(aix_layer.alpha());
//   } else {
//     layer.alpha(std::numeric_limits<float>::quiet_NaN());
//   }

//   if (aix_layer.has_gamma()) {
//     layer.gamma(aix_layer.gamma());
//   } else {
//     layer.gamma(std::numeric_limits<float>::quiet_NaN());
//   }


//   if (aix_layer.has_axis()) {
//     layer.axis(aix_layer.axis());
//   } else {
//     layer.axis(std::numeric_limits<int>::lowest());
//   }

//   if (aix_layer.has_stash_type()) {
//     layer.stash_type(aix_layer.stash_type());
//   } else {
//     layer.stash_type(std::numeric_limits<int>::lowest());
//   }

//   if (aix_layer.has_beta()) {
//     layer.beta(aix_layer.beta());
//   } else {
//     layer.beta(std::numeric_limits<float>::quiet_NaN());
//   }

//   if (aix_layer.has_transa()) {
//     layer.trans_A(aix_layer.transa());
//   } else {
//     layer.trans_A(std::numeric_limits<int64_t>::lowest());
//   }

//   if (aix_layer.has_transb()) {
//     layer.trans_B(aix_layer.transb());
//   } else {
//     layer.trans_B(std::numeric_limits<int64_t>::lowest());
//   }

//   if (aix_layer.has_keepdims()) {
//     layer.keepdims(aix_layer.keepdims());
//   } else {
//     layer.keepdims(std::numeric_limits<int64_t>::lowest());
//   }


//   if (aix_layer.has_noop_with_empty_axes()) {
//     layer.noop_with_empty_axes(aix_layer.noop_with_empty_axes());
//   } else {
//     layer.noop_with_empty_axes(std::numeric_limits<int64_t>::lowest());
//   }

//   if (aix_layer.has_select_last_index()) {
//     layer.select_last_index(aix_layer.select_last_index());
//   } else {
//     layer.select_last_index(std::numeric_limits<int64_t>::lowest());
//   }

//   // Parse axes
//   layer.axes(vector<int64_t>(aix_layer.axes().size()));
//   auto &axes = layer.axes();
//   for (int i = 0; i < aix_layer.axes().size(); ++i) {
//     axes.at(i) = aix_layer.axes(i);
//   }

//   if (aix_layer.has_mode()) {
//     layer.mode(aix_layer.mode());
//   }

//   if (aix_layer.has_extrapolation_value()) {
//     layer.extrapolation_value(aix_layer.extrapolation_value());
//   } else {
//     layer.extrapolation_value(std::numeric_limits<float>::quiet_NaN());
//   }

//   if (aix_layer.has_coordinate_transformation_mode()) {
//     layer.coordinate_transformation_mode(
//         aix_layer.coordinate_transformation_mode());
//   }

//   if (aix_layer.has_nearest_mode()) {
//     layer.nearest_mode(aix_layer.nearest_mode());
//   }

//   if (aix_layer.has_cubic_coeff_a()) {
//     layer.cubic_coeff_a(aix_layer.cubic_coeff_a());
//   } else {
//     layer.cubic_coeff_a(std::numeric_limits<float>::quiet_NaN());
//   }

//   if (aix_layer.has_exclude_outside()) {
//     layer.exclude_outside(aix_layer.exclude_outside());
//   } else {
//     layer.exclude_outside(std::numeric_limits<int64_t>::lowest());
//   }

//   if (aix_layer.has_auto_pad()) {
//     layer.auto_pad(aix_layer.auto_pad());
//   }

//   if (aix_layer.has_group()) {
//     layer.group(aix_layer.group());
//   }

//   // Handle dilations
//   layer.dilations(vector<int64_t>(aix_layer.dilations().size()));
//   auto &dilations = layer.dilations();
//   for (int i = 0; i < aix_layer.dilations().size(); ++i) {
//     dilations.at(i) = aix_layer.dilations(i);
//   }

//   // Handle kernel shapes
//   layer.kernel_shape(vector<int64_t>(aix_layer.kernel_shape().size()));
//   auto &kernel_shape = layer.kernel_shape();
//   for (int i = 0; i < aix_layer.kernel_shape().size(); ++i) {
//     kernel_shape.at(i) = aix_layer.kernel_shape(i);
//   }

//   // Handle output padding
//   layer.output_padding(vector<int64_t>(aix_layer.output_padding().size()));
//   auto &output_padding = layer.output_padding();
//   for (int i = 0; i < aix_layer.output_padding().size(); ++i) {
//     output_padding.at(i) = aix_layer.output_padding(i);
//   }

//   // Handle output shape
//   layer.output_shape(vector<int64_t>(aix_layer.output_shape().size()));
//   auto &output_shape = layer.output_shape();
//   for (int i = 0; i < aix_layer.output_shape().size(); ++i) {
//     output_shape.at(i) = aix_layer.output_shape(i);
//   }

//   // Handle padding
//   layer.pads(vector<int64_t>(aix_layer.pads().size()));
//   auto &pads = layer.pads();
//   for (int i = 0; i < aix_layer.pads().size(); ++i) {
//     pads.at(i) = aix_layer.pads(i);
//   }


//   // Handle strides
//   layer.strides(vector<int64_t>(aix_layer.strides().size()));
//   auto &strides = layer.strides();
//   for (int i = 0; i < aix_layer.strides().size(); ++i) {
//     strides.at(i) = aix_layer.strides(i);
//   }

//   // Handle thresholds
//   layer.input_thresholds(vector<float>(1));
//   auto &input_thresholds = layer.input_thresholds();
//   input_thresholds.at(0) = aix_layer.input_threshold();

//   if (aix_layer.has_output_threshold()) {
//     layer.output_threshold(aix_layer.output_threshold());
//   }


//   layer.filter_thresholds(vector<float>(aix_layer.filter_threshold().size()));
//   auto &filter_thresholds = layer.filter_thresholds();
//   for (int i = 0; i < aix_layer.filter_threshold().size(); ++i) {
//     filter_thresholds.at(i) = aix_layer.filter_threshold(i);
//   }
  
//   LOG(INFO) << "-------->>>  Finished parsing layer types.";

//   return layer;
// }


shared_ptr<Tensor> SCOPE::ParseAIXTensor(const AIXTensor &aix_tensor,
                                         const int group) {
  DataType dtype = static_cast<DataType>(aix_tensor.dtype());
  shared_ptr<Tensor> tensor;

  if (aix_tensor.dims().size() == 1) {
    tensor = std::make_shared<Tensor>(aix_tensor.dims(0), dtype);
  } else if (aix_tensor.dims().size() == 4) {
    int channel = aix_tensor.dims(2);
    if (group != 1) {
      channel /= group;
    }
    tensor =
        std::make_shared<Tensor>(aix_tensor.dims(3), channel,
                                 aix_tensor.dims(1), aix_tensor.dims(0), dtype);
  } else {
    auto dims = vector<size_t>(aix_tensor.dims().size());
    for (int i = 0; i < aix_tensor.dims().size(); ++i) {
      dims.at(i) = static_cast<size_t>(aix_tensor.dims(i));
    }
    tensor = std::make_shared<Tensor>(std::move(dims), dtype);
  }

  void *data = tensor->data();

  switch (dtype) {
    case dty::DataType::FP32:
      memcpy(data, aix_tensor.fval().data(), tensor->size());
      break;
    case dty::DataType::INT8:
    case dty::DataType::UINT8:
      memcpy(data, aix_tensor.bval().data(), tensor->size());
      break;
    case dty::DataType::INT16:
    case dty::DataType::FP64:
    case dty::DataType::FP16:
    default:
      LOG(ERROR) << "Not Implemented Data Type! : " << dtype << "\n";
      exit(1);
      break;
  }

  return tensor;
}

std::shared_ptr<Descriptor> SCOPE::ParseAIXConvolutionDesc(
    const AIXConvolutionDesc &convdesc) {
  auto desc = make_shared<Descriptor>();

  desc->padding_width_right(convdesc.padding(0));
  desc->padding_width_left(convdesc.padding(1));
  desc->padding_height_bottom(convdesc.padding(2));
  desc->padding_height_top(convdesc.padding(3));

  desc->stride_height(convdesc.stride(0));
  desc->stride_width(convdesc.stride(1));


  desc->dilation_width(convdesc.dilation(1));
  desc->dilation_height(convdesc.dilation(3));

  desc->groups(convdesc.groups());

  return desc;
}

std::shared_ptr<Descriptor> SCOPE::ParseAIXEWAddDesc(
    const AIXEWAddDesc &ewadddesc) {
  auto desc = make_shared<Descriptor>();
  desc->scale(ewadddesc.scale(0));
  return desc;
}

std::shared_ptr<Descriptor> SCOPE::ParseAIXSamplingDesc(
    const AIXSamplingDesc &samplingdesc) {
  auto desc = make_shared<Descriptor>();

  desc->window_height(samplingdesc.window(0));
  desc->window_width(samplingdesc.window(1));

  desc->stride_height(samplingdesc.stride(0));
  desc->stride_width(samplingdesc.stride(1));

  desc->padding_width_right(samplingdesc.padding(0));
  desc->padding_width_left(samplingdesc.padding(1));
  desc->padding_height_bottom(samplingdesc.padding(2));
  desc->padding_height_top(samplingdesc.padding(3));

  return desc;
}

expected<void, SimulatorError> SCOPE::DumpGraphBinary(
    const string &binary_path) {
  fstream new_aix_graph(binary_path.c_str(), ios::out | ios::binary);
  if (!new_aix_graph) {
    const string msg = "Failed to open file: `" + binary_path + "`";
    LOG(ERROR) << msg;
    return unexpected<SimulatorError>(SimulatorError::kFileWriteError);
  }

  bool result = aix_graph_.SerializeToOstream(&new_aix_graph);
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

  const float input_threshold = network->layers(0).input_thresholds(0);
  aix_graph_.mutable_layer(0)->set_input_threshold(input_threshold);

  for (size_t i = 0; i < num_layers; ++i) {
    const float output_threshold = network->layers(i).output_threshold();
    aix_graph_.mutable_layer(i)->set_output_threshold(output_threshold);
    const auto filter_thresholds = network->layers(i).filter_thresholds();
    for (auto idx = 0; idx < filter_thresholds.size(); idx++) {
      aix_graph_.mutable_layer(i)->set_filter_threshold(
          idx, filter_thresholds.at(idx));
    }
  }
  return {};
}
