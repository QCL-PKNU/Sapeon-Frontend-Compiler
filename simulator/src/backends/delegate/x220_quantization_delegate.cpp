#include "backends/delegate/x220_quantization_delegate.hpp"

#include <algorithm>
#include <cassert>
#include <memory>
#include <set>
#include <string>

#include "arguments.hpp"
#include "backends/backend.hpp"
#include "enums/error.hpp"
#include "factory.hpp"
#include "glog/logging.h"
#include "network/network.hpp"
#include "tl/expected.hpp"
#include "utility.hpp"
#include "x220/aixh_common.h"
#include "x220/ops/x220_operation.hpp"
#include "x220/quant_config.hpp"

namespace quantization {

X220QuantizationDelegate::X220QuantizationDelegate(Backend &parent,
                                                   Arguments &args)
    : parent_(parent), dump_(args) {}

tl::expected<void, SimulatorError> X220QuantizationDelegate::Quantize(
    std::unique_ptr<Network> &network) {
  LOG(INFO) << "Quantize Started\n";
  struct timespec start_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);

  LOG(INFO) << "InitQuantConfigDataType Started";
  InitQuantConfigDataType(network);
  LOG(INFO) << "InitQuantConfigDataType Finished";
  LOG(INFO) << "PromoteQuantConfigDataType Started";
  PromoteQuantConfigDataType(network);
  LOG(INFO) << "PromoteQuantConfigDataType Finished";
  LOG(INFO) << "SetQuantConfig Started";
  SetQuantConfig(network);
  LOG(INFO) << "SetQuantConfig Finished";

  // Dump weights, biases
  // Need to implement
  dump_.DumpX220QuantizedNetworkInfo(network);
  PrintElapsedTime(start_time);
  LOG(INFO) << "Quantize Finished\n";
  return {};
}

void X220QuantizationDelegate::SetQuantConfig(
    std::unique_ptr<Network> &network) {
  const int num_layers = network->num_layers();
  for (int i = 0; i < num_layers; i++) {
    Layer &layer = network->layers(i);
    const int num_sublayers = network->num_operations(i);
    for (int j = 0; j < num_sublayers; j++) {
      std::string operation_name = layer.operation_types(j);

      auto quant_op =
          Factory<x220::X220Operation>::CreateInstance(operation_name);
      if (quant_op == nullptr) {
        continue;
      }
      quant_op->PrepareQuantOperation(network, i);
    }
  }
}

void X220QuantizationDelegate::InitQuantConfigDataType(
    std::unique_ptr<Network> &network) {
  for (auto &layer : network->layers()) {
    x220::QuantConfig &config = layer.x220_quant_config();
    config.in_dtype(x220::DataType::DTY_SINT8);
    config.out_dtype(x220::DataType::DTY_SINT8);
  }
}

void X220QuantizationDelegate::PromoteQuantConfigDataType(
    std::unique_ptr<Network> &network) {
  const int num_layers = network->num_layers();
  for (int i = 0; i < num_layers; i++) {
    TryPromotion(network, i);
  }

  bool converged = false;
  while (!converged) {
    converged = true;
    for (int i = 0; i < num_layers; i++) {
      converged &= UpdateDataType(network, i);
    }
  }
}

bool X220QuantizationDelegate::UpdateDataType(std::unique_ptr<Network> &network,
                                              const int idx_layer) {
  bool converged = true;
  auto &layer = network->layers(idx_layer);
  auto &config = layer.x220_quant_config();
  const auto &predecessors = layer.predecessors();

  // inherit data type from predecessor
  for (const auto pred : predecessors) {
    auto &pred_layer = network->layers(pred);
    auto &pred_config = pred_layer.x220_quant_config();
    if (pred_config.out_dtype() != config.in_dtype()) {
      config.in_dtype(pred_config.out_dtype());
      converged = false;
    }
  }
  // non- conv/connected layers need consistent i_dtype and o_dtype
  if (!CheckConvOrConnected(layer) && config.out_dtype() != config.in_dtype()) {
    config.out_dtype(config.in_dtype());
    converged = false;
  }
  return converged;
}

void X220QuantizationDelegate::TryPromotion(std::unique_ptr<Network> &network,
                                            const int idx_layer) {
  auto &layer = network->layers(idx_layer);
  // output data type promotion
  auto empty_set = std::set<int>{};
  if (AllowOutputPromotion(empty_set, network, layer)) {
    std::set<int> visited;
    visited.insert(layer.id());
    bool allow_promotion = true;
    allow_promotion = AllowSuccessorInputPromotion(visited, network, layer);

    if (allow_promotion) {
      auto &config = layer.x220_quant_config();
      config.out_dtype(x220::DataType::DTY_UINT8);
    }
  }
}

bool X220QuantizationDelegate::AllowOutputPromotion(
    std::set<int> &visited, std::unique_ptr<Network> &network, Layer &layer) {
  if (CheckConvLayer(layer)) {
    return CheckConvOrConnected(layer) && CheckNegSlope(layer);
  }

  visited.insert(layer.id());
  bool allow_promotion = true;
  // check if preds allow promotion
  allow_promotion = AllowPredecessorOutputPromotion(visited, network, layer);

  // if multiple succs, check if the other branches support promotion
  const auto &succs = layer.successors();
  if (allow_promotion && succs.size() > 1) {
    allow_promotion = AllowSuccessorInputPromotion(visited, network, layer);
  }
  return allow_promotion;
}

bool X220QuantizationDelegate::AllowInputPromotion(
    std::set<int> &visited, std::unique_ptr<Network> &network, Layer &layer) {
  if (CheckConvLayer(layer)) {
    return CheckConvOrConnected(layer);
  }

  visited.insert(layer.id());
  bool allow_promotion = true;
  // check if succs allow promotion
  allow_promotion = AllowSuccessorInputPromotion(visited, network, layer);

  // if multiple preds, check if the other branches support promotion
  const auto &preds = layer.predecessors();
  if (allow_promotion && preds.size() > 1) {
    allow_promotion = AllowPredecessorOutputPromotion(visited, network, layer);
  }
  return allow_promotion;
}

bool X220QuantizationDelegate::AllowSuccessorInputPromotion(
    std::set<int> &visited, std::unique_ptr<Network> &network, Layer &layer) {
  bool allow_promotion = true;
  const auto &succs = layer.successors();
  auto succ = succs.begin();  // succs_: list
  while (allow_promotion && succ != succs.end()) {
    auto &succ_layer = network->layers(*succ);
    if (visited.find(succ_layer.id()) != visited.end()) {
      allow_promotion = AllowInputPromotion(visited, network, succ_layer);
    }
    ++succ;
  }
  return allow_promotion;
}

bool X220QuantizationDelegate::AllowPredecessorOutputPromotion(
    std::set<int> &visited, std::unique_ptr<Network> &network, Layer &layer) {
  bool allow_promotion = true;
  const auto &preds = layer.predecessors();
  auto pred = preds.begin();  // preds_: list
  while (allow_promotion && pred != preds.end()) {
    auto &pred_layer = network->layers(*pred);
    if (visited.find(pred_layer.id()) != visited.end()) {
      allow_promotion = AllowOutputPromotion(visited, network, pred_layer);
    }
    ++pred;
  }
  return allow_promotion;
}

bool X220QuantizationDelegate::CheckConvLayer(Layer &layer) {
  bool isConvLayer = false;
  const auto &operations = layer.operation_types();
  for (auto &operation_name : operations) {
    isConvLayer |= operation_name == "Convolution";
    isConvLayer |= operation_name == "Connected";
    isConvLayer |= operation_name == "GroupConvolution";
  }
  return isConvLayer;
}

bool X220QuantizationDelegate::CheckConvOrConnected(Layer &layer) {
  bool isConvOrConnectedLayer = false;
  const auto &operations = layer.operation_types();
  for (auto &operation_name : operations) {
    isConvOrConnectedLayer |= operation_name == "Convolution";
    isConvOrConnectedLayer |= operation_name == "Connected";
  }
  return isConvOrConnectedLayer;
}

bool X220QuantizationDelegate::CheckNegSlope(Layer &layer) {
  bool isNegSlopeZero = false;
  isNegSlopeZero |= layer.activation_type() == "ReLU";
  isNegSlopeZero |= layer.activation_type() == "ReLU6";
  return isNegSlopeZero;
}
}  // namespace quantization
