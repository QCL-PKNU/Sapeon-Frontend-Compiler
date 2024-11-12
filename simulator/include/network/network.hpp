#ifndef NETWORK_NETWORK_HPP
#define NETWORK_NETWORK_HPP

#include <memory>
#include <string>
#include <vector>

#include "network/layer.hpp"
#include "network/tensor.hpp"

class Network {
 public:
  Network();
  Network(std::vector<Layer> layers, std::vector<int> num_operations,
          const std::string &graph_type);

  bool CheckValidNetwork(const std::string &backend_type, bool do_quant);
  void PostProcessCalibration();

  int num_layers();
  void num_layers(int value);
  void num_operations(std::vector<int> num_operations);
  std::vector<int> &num_operations();
  int num_operations(int idx);
  int num_outputs();
  void num_outputs(int value);
  void input_layer(Layer layer) { input_layer_ = layer; };
  Layer &input_layer() { return input_layer_; }
  void layers(std::vector<Layer> layers);
  std::vector<Layer> &layers();
  Layer &layers(int idx);
  void graph_type(const std::string &graph_type);
  const std::string &graph_type();

 private:
  void SetRouteThreshold();
  int num_layers_;
  int num_outputs_;
  std::vector<int> num_operations_;
  std::vector<Layer> layers_;
  Layer input_layer_;
  std::string graph_type_;
};

#endif  // NETWORK_NETWORK_HPP
