#include "x220/ops/x220_operation.hpp"

#define CLASS X220Operation
#define SCOPE CLASS

#include <memory>
using std::make_unique;
using std::unique_ptr;

#include "glog/logging.h"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "x220/aixh_common.h"
#include "x220/quant_config.hpp"

namespace x220 {
void SCOPE::PrepareQuantOperation(unique_ptr<Network>& network,
                                  const int idx_layer) {
  InitQuantConfig(network, idx_layer);
  QuantizeLayer(network->layers(idx_layer));
}

void SCOPE::CalculateMultiplierShifter(DataType dtype, float threshold_ratio,
                                       int& multiplier, int& shifter) {
  int fixed_min, fixed_max, shift_amount;

  if (dtype == DataType::DTY_SINT8) {
    shift_amount = 15;
    fixed_max = (0x1 << 7);
  } else if (dtype == DataType::DTY_SINT16) {
    shift_amount = 31;
    fixed_max = (0x1 << 15);
  } else if (dtype == DataType::DTY_UINT8) {
    shift_amount = 15;
    fixed_max = (0x1 << 8);
  }
  fixed_min = -fixed_max;

  int64_t scale = (0x1LL << shift_amount);
  int64_t fixed_ratio =
      static_cast<int>(std::nearbyint(threshold_ratio * scale));
  while (fixed_ratio >= fixed_max || fixed_ratio < fixed_min) {
    --shift_amount;
    scale >>= 1;
    fixed_ratio = static_cast<int64_t>(
        std::nearbyint(threshold_ratio * static_cast<double>(scale)));
  }

  multiplier = fixed_ratio;
  shifter = shift_amount;
}

void SCOPE::InitCommonQuantConfig(unique_ptr<Network>& network,
                                  const int idx_layer) {
  auto& layer = network->layers(idx_layer);
  x220::QuantConfig& config = layer.x220_quant_config();

  config.wquant_en(1);    // remove
  config.aquant_en(1);    // remove
  config.wamplifier(1);   // remove
  config.calib_mode(14);  // remove
  config.calib_adj(1);    // remove

  float input_threshold;
  if (idx_layer == 0) {
    input_threshold = layer.input_thresholds(0);
  } else {
    Layer& prev_layer = network->layers(layer.predecessors(0));
    input_threshold = prev_layer.output_threshold();
  }

  float threshold_ratio = input_threshold / layer.output_threshold();
  int multiplier, shifter;
  CalculateMultiplierShifter(config.out_dtype(), threshold_ratio, multiplier,
                             shifter);
  config.multiplier(multiplier);
  config.shifter(shifter);

  int quant_min, quant_max;
  x220::GetDataTypeMinMax(quant_min, quant_max, config.out_dtype());

  int quant_in_min, quant_in_max;
  x220::GetDataTypeMinMax(quant_in_min, quant_in_max, config.in_dtype());

  // TODO: check mxc task and update quant_min in each operation
  config.iscale(static_cast<float>(quant_in_max) / input_threshold);
  config.oscale(static_cast<float>(quant_max) / layer.output_threshold());
  config.oqbias(0);
  config.oqmax(static_cast<float>(quant_max));
  config.oqmin(static_cast<float>(quant_min));
}
}  // namespace x220
