#ifndef ENUMS_OPERATION_NAMES_HPP
#define ENUMS_OPERATION_NAMES_HPP

#include <array>
#include <string>
#include <unordered_map>

#include "glog/logging.h"

static std::unordered_map<int, std::string> AIX_OPERATIONS = {
    // AIXLayerType, operation class name
    // check proto_files/aixh.proto
    {0, "Convolution"},
    {1, "Connected"},
    {2, "Maxpool"},
    {3, "Avgpool"},
    {6, "Route"},
    {7, "Reorg"},
    {8, "EWAdd"},
    {9, "Upsample"},
    {10, "Pixelshuffle"},
    {11, "GroupConvolution"},
    {12, "SkipConvolution"},
    {13, "Activations"},
    {14, "BatchNormalization"},
    {15, "BiasAdd"},
    {16, "Output"},
    {17, "Input"},
    // {18, "Wildcard"},
    {19, "Add"},
    {20, "Mul"},
    {21, "Sub"},
    // {22, "Sum"},
    // {23, "InstanceNorm"},
    // {24, "MatMul"},
    // {25, "LayerNorm"},
};

static std::unordered_map<std::string, std::string> SPEAR_OPERATIONS{
    // layer_type, operation class name
    // check proto_files/layer_type.txt
    {"convolution", "Convolution"},
    {"connected", "Connected"},
    {"maxpool", "Maxpool"},
    {"gavgpool", "Avgpool"},
    {"softmax", "Softmax"},
    {"route", "Route"},
    {"reorg", "Reorg"},
    {"ewadd", "EWAdd"},
    {"upsample", "Upsample"},
    {"bilinear-upsample", "BilinearUpsample"},
    {"pixelshuffle", "Pixelshuffle"},
    {"groupconv", "GroupConvolution"},
    {"activation", "Activations"},
    {"batchnorm", "BatchNormalization"},
    {"biasadd", "BiasAdd"},
    // {"wildcard", ""},
    {"lavgpool", "Lavgpool"},
    {"ewmul", "EWMul"},
    // {"move", ""},
};

static std::string GetOperationName(const std::string &key,
                                    const std::string &graph_type) {
  if (graph_type == "aix_graph") {
    const auto idx = std::stoi(key);
    if (idx >= AIX_OPERATIONS.size()) {
      LOG(ERROR) << "Undefined AIXOperation key number! : " << idx << "\n";
      exit(1);
    }
    std::string ops_name = AIX_OPERATIONS[idx];
    return ops_name;
  } else if (graph_type == "spear_graph") {
    auto it = SPEAR_OPERATIONS.find(key);
    std::string ops_name = "";
    if (it != SPEAR_OPERATIONS.end()) ops_name = SPEAR_OPERATIONS.at(key);
    return ops_name;
  } else {
    LOG(ERROR) << "Undefined Graph Type! : " << graph_type << "\n";
    exit(1);
  }
}

#endif  // ENUMS_OPERATION_NAMES_HPP
