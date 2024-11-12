#include "network/network.hpp"

#include <memory>
#include <queue>
#include <string>
#include <vector>

#ifdef GPU
#include <cudnn.h>
#endif

#include "factory.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cpu_operation.hpp"
#include "parser/parser.hpp"
#include "x330/x330_operation.hpp"

#ifdef GPU
#include "operations/cudnn_operation.hpp"
#endif

Network::Network() { input_layer_.id(-1);
  LOG(INFO) << "Default Network constructor called";
}

Network::Network(std::vector<Layer> layers, std::vector<int> num_operations,
                 const std::string &graph_type)
    : num_layers_(layers.size()),
      layers_(layers),
      num_operations_(num_operations),
      graph_type_(graph_type) {
  // TODO(Wooseong) : initialize input_layer_ in parser
  input_layer_.id(-1);
  LOG(INFO) << "Parameterized Network constructor called";
}



bool Network::CheckValidNetwork(const std::string &backend_type,
                                bool do_quant) {

  LOG(INFO) << "Total layers in the network: " << layers_.size();

  int new_id = 0;
  for (auto &layer : layers_) {
    layer.id(new_id);  // Assuming `id()` is a setter function to assign the layer's ID
    new_id++;  // Increment the ID for the next layer
  }

  int i = 0;
  for (auto &layer : layers_) {

    if (!layer.CheckValidLayer(backend_type, do_quant)) {
      DLOG(ERROR) << "layer " << layer.id() << " (" << layer.name() << ")"
                  << " is invalid";
      return false;
    }

    if (i == 171) {
      break;
    }
    i++;
  }

  return true;
}

void Network::PostProcessCalibration() { SetRouteThreshold(); }

int Network::num_layers() { return num_layers_; }

void Network::num_layers(int value) { num_layers_ = value; }

void Network::num_operations(std::vector<int> num_operations) {
  num_operations_ = num_operations;
}

std::vector<int> &Network::num_operations() { return num_operations_; }

int Network::num_operations(int idx) { return num_operations_.at(idx); }

int Network::num_outputs() { return num_outputs_; }

void Network::num_outputs(int value) { num_outputs_ = value; }

// TODO : check this if it has been called
void Network::layers(std::vector<Layer> layers) { layers_ = layers; }

std::vector<Layer> &Network::layers() { return layers_; }

Layer &Network::layers(int idx) { return layers_.at(idx); }

void Network::graph_type(const std::string &graph_type) {
  graph_type_ = graph_type;
}

const std::string &Network::graph_type() { return graph_type_; }

void Network::SetRouteThreshold() {
  for (size_t idx_layer = 0; idx_layer < num_layers_; ++idx_layer) {
    Layer &layer = layers_.at(idx_layer);

    if (layer.HasOperationTypes("Route")) {
      const size_t pred_size = layer.predecessors().size();
      float max_threshold = std::numeric_limits<float>::lowest();

      for (int idx_pred = 0; idx_pred < pred_size; ++idx_pred) {
        const int pred_id = layer.predecessors(idx_pred);
        Layer &pred_layer = layers_.at(pred_id);
        max_threshold = std::max(max_threshold, pred_layer.output_threshold());
      }

      for (int idx_pred = 0; idx_pred < pred_size; ++idx_pred) {
        const int pred_id = layer.predecessors(idx_pred);
        Layer &pred_layer = layers_.at(pred_id);
        pred_layer.output_threshold(max_threshold);

        std::queue<std::reference_wrapper<Layer>> route_preds_q;

        if (pred_layer.HasOperationTypes("Route")) {
          route_preds_q.push(pred_layer);
        }

        while (!route_preds_q.empty()) {
          Layer &temp_layer = route_preds_q.front();
          route_preds_q.pop();
          const int temp_pred_size = temp_layer.predecessors().size();
          for (int idx_temp_pred = 0; idx_temp_pred < temp_pred_size;
               ++idx_temp_pred) {
            const int temp_pred_id = temp_layer.predecessors(idx_temp_pred);
            Layer &pred_temp_layer = layers_.at(temp_pred_id);
            pred_temp_layer.output_threshold(max_threshold);

            if (pred_temp_layer.HasOperationTypes("Route")) {
              route_preds_q.push(pred_temp_layer);
            }
          }
        }
      }
      layer.output_threshold(max_threshold);
    }
  }
}
