#ifndef ENUMS_ACTIVATION_NAMES_HPP
#define ENUMS_ACTIVATION_NAMES_HPP

#include <array>
#include <string>

#include "glog/logging.h"

static std::array<std::string, 10> AIX_ACTIVATIONS = {
    "Sigmoid",  "ReLU", "LeakyReLU", "PReLU", "Tanh",
    "Identity", "Mish", "Celu",      "Selu",  "Softmax"};

static std::array<std::string, 10> SPEAR_ACTIVATIONS = {
    "Sigmoid",  "ReLU", "LeakyReLU", "PReLU", "Tanh",
    "Identity", "Mish", "ReLU6",     "Swish", "CWPrelu"};

static std::string GetActivationName(int idx, const std::string &graph_type) {
  if (graph_type == "aix_graph") {
    if (idx >= AIX_ACTIVATIONS.size()) {
      LOG(ERROR) << "Undefined AIXActivation key number! : " << idx << "\n";
      exit(1);
    }
    std::string ops_name = AIX_ACTIVATIONS[idx];
    return ops_name;
  } else if (graph_type == "spear_graph") {
    if (idx >= SPEAR_ACTIVATIONS.size()) {
      LOG(ERROR) << "Undefined SpearActivation key number! : " << idx << "\n";
      exit(1);
    }
    std::string ops_name = SPEAR_ACTIVATIONS[idx];
    return ops_name;
  } else {
    LOG(ERROR) << "Undefined Graph Type! : " << graph_type << "\n";
    exit(1);
  }
}

#endif  // ENUMS_ACTIVATION_NAMES_HPP
