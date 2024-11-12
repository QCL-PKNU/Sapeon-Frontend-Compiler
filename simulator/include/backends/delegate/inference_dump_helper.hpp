#ifndef BACKENDS_DELEGATE_INFERENCE_DUMP_HELPER_HPP
#define BACKENDS_DELEGATE_INFERENCE_DUMP_HELPER_HPP

#include <memory>
#include <string>

#include "arguments.hpp"
#include "dump/dump_factory.hpp"
#include "dump/dump_path_helper.hpp"
#include "enums/dump.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"

class InferenceDumpHelper {
 public:
  InferenceDumpHelper(Arguments &args);
  void DumpLayerOutput(std::shared_ptr<Tensor> activation, int idx_layer,
                       DumpLevel dump_level);
  void DumpNetworkOutput(std::shared_ptr<Tensor> output, DumpLevel dump_level);
  void DumpX220Activation(std::shared_ptr<Tensor> activation, int idx_layer,
                          DumpLevel dump_level, float output_threshold);
  void DumpX220Input(std::shared_ptr<Tensor> tensor, DumpLevel dump_level,
                     float input_threshold);
  void DumpNetworkInput(std::shared_ptr<Tensor> tensor, DumpLevel dump_level);
  void DumpQuantizedInput(std::shared_ptr<Tensor> input, double input_scale);
  void DumpX330NetworkInput(std::shared_ptr<Tensor> tensor);
  void DumpX330LayerOutput(const Tensor &output_tensor, int idx_layer);

 private:
  template <typename Type>
  void DumpLayerOutput(std::shared_ptr<Tensor> activation, int idx_layer,
                       DumpLevel dump_level);
  template <typename Type>
  void DumpNetworkOutput(std::shared_ptr<Tensor> output, DumpLevel dump_level);
  template <typename Type>
  void DumpNetworkInput(std::shared_ptr<Tensor> output, DumpLevel dump_level);
  template <typename Type>
  void DumpX220Activation(std::shared_ptr<Tensor> activation, int idx_layer,
                          DumpLevel dump_level, float output_threshold);
  template <typename Type>
  void DumpTensorDefault(const std::string &file_name,
                         std::shared_ptr<Tensor> tensor);
  template <typename Type>
  void DumpTensorDarknet(const std::string &file_name,
                         std::shared_ptr<Tensor> tensor);
  template <typename Type>
  void DumpTensorDarknetDequant(const std::string &file_name,
                                std::shared_ptr<Tensor> tensor, double scale);
  DumpPathHelper path_helper_;
  DumpFactory factory_;
  DumpLevel dump_level_;
};

#endif  // BACKENDS_DELEGATE_INFERENCE_DUMP_HELPER_HPP
