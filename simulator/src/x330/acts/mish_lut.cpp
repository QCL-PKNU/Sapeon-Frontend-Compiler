#include "x330/acts/mish_lut.hpp"

#include <algorithm>
#include <memory>

#include "factory.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"
#include "network/tensor.hpp"
#include "x330/aixv_base.h"
#include "x330/aixv_float.h"

namespace x330 {
namespace {
class AixLutModel {
 public:
  AixLutModel(const std::string& cfg_fname) : cfg_fname_(cfg_fname) {}
  ~AixLutModel() {}

  void Init() {
    FILE* fp = fopen(cfg_fname_.c_str(), "rt");
    AIXV_CHECK(fp != nullptr, "FAILED to open: %s", cfg_fname_);

    char line[1024];

    //
    // Header
    //
    bool found_header = false;
    bool found_range = false;
    bool found_lut0_size = false;
    bool found_lut1_size = false;

    while (!found_header && fgets(line, sizeof(line), fp)) {
      {
        const char* prefix = "# range = (min, max) = ";
        const int prefix_len = strlen(prefix);
        if (strncmp(prefix, line, prefix_len) == 0) {
          AIXV_CHECK_EQ(
              sscanf(line + prefix_len, "(%f, %f)", &range_lb_, &range_ub_), 2,
              "<%s> unexpected pattern: %s\n", cfg_fname_, line);
          found_range = true;
        }
      }
      {
        const char* prefix = "# num_lut0_used = ";
        const int prefix_len = strlen(prefix);
        if (strncmp(prefix, line, prefix_len) == 0) {
          AIXV_CHECK_EQ(sscanf(line + prefix_len, "%d", &lut0_size_), 1,
                        "<%s> unexpected pattern: %s\n", cfg_fname_, line);
          found_lut0_size = true;
        }
      }
      {
        const char* prefix = "# num_lut1_used = ";
        const int prefix_len = strlen(prefix);
        if (strncmp(prefix, line, prefix_len) == 0) {
          AIXV_CHECK_EQ(sscanf(line + prefix_len, "%d", &lut1_size_), 1,
                        "<%s> unexpected pattern: %s\n", cfg_fname_, line);
          found_lut1_size = true;
        }
      }
      found_header = found_range && found_lut0_size && found_lut1_size;
    }

    AIXV_CHECK(found_range, "<%s> range not found\n", cfg_fname_);
    AIXV_CHECK(found_lut0_size, "<%s> LUT0 size not found\n", cfg_fname_);
    AIXV_CHECK(found_lut1_size, "<%s> LUT1 size not found\n", cfg_fname_);

    //
    // LUT contents
    //
    lut0_entries_.resize(lut0_size_);
    lut1_entries_.resize(lut1_size_ + 2 /*extra entries for out-of-range*/);

    int lut_level = -1;
    while (fgets(line, sizeof(line), fp)) {
      {
        const char* prefix = "- lut";
        const int prefix_len = strlen(prefix);
        if (strncmp(prefix, line, prefix_len) == 0) {
          AIXV_CHECK_EQ(sscanf(line + prefix_len, "%d", &lut_level), 1,
                        "<%s> unexpected pattern: %s\n", cfg_fname_, line);
          continue;
        }
      }

      {
        const char* prefix = "  [";
        const int prefix_len = strlen(prefix);
        if (strncmp(prefix, line, prefix_len) != 0) {
          continue;
        }

        int idx;
        char lb_str[128];
        char ub_str[128];
        float slope, offset;
        int lut1_sofs, lut1_segs;

        if (lut_level == 0) {
          AIXV_CHECK_EQ(
              sscanf(line + prefix_len, "%d] %s %s : %f %f %d %d", &idx, lb_str,
                     ub_str, &slope, &offset, &lut1_sofs, &lut1_segs),
              7, "<%s> unexpected LUT entry format: %s", cfg_fname_, line);
          AIXV_CHECK_GE(idx, 0);
          AIXV_CHECK_LT(idx, lut0_size_);
          AIXV_CHECK_GE(lut1_sofs, -1);
          AIXV_CHECK_LE(lut1_sofs + lut1_segs, lut1_size_);

          auto& e = lut0_entries_[idx];
          e.slope = NF16::FromFloat(slope);
          e.offset = NF16::FromFloat(offset);
          e.lut1_sofs = lut1_sofs;
          e.lut1_segs = lut1_segs;
        } else if (lut_level == 1) {
          AIXV_CHECK_EQ(sscanf(line + prefix_len, "%d] %s %s : %f %f", &idx,
                               lb_str, ub_str, &slope, &offset),
                        5, "<%s> unexpected LUT entry format: %s", cfg_fname_,
                        line);
          AIXV_CHECK_GE(idx, 0);
          AIXV_CHECK_LT(idx, lut1_size_ + 2);

          auto& e = lut1_entries_[idx];
          e.slope = NF16::FromFloat(slope);
          e.offset = NF16::FromFloat(offset);
        }
      }
    }  // while

    fclose(fp);
    initialized_ = true;
  }

  void Activate(float* data_ptr, int nelems) {
    if (!initialized_) Init();

    const float lut0_step_size = (range_ub_ - range_lb_) / lut0_size_;

#pragma omp parallel for
    for (int i = 0; i < nelems; ++i) {
      float in_value = data_ptr[i];

      LutEntry entry;
      if (in_value < range_lb_) {
        entry = lut1_entries_[lut1_size_ + 0];
      } else if (in_value >= range_ub_) {
        entry = lut1_entries_[lut1_size_ + 1];
      } else {
        float in_shifted = in_value - range_lb_;
        int lut0_idx = in_shifted / lut0_step_size;
        entry = lut0_entries_[lut0_idx];

        if (entry.lut1_sofs >= 0) {
          in_shifted -= lut0_step_size * lut0_idx;
          int lut1_idx =
              entry.lut1_sofs + entry.lut1_segs * in_shifted / lut0_step_size;
          entry = lut1_entries_[lut1_idx];
        }
      }

#if 1
      if (calc_mode_ == LutCalcMode::FP32) {
        float slope = entry.slope.ToFloat();
        float offset = entry.offset.ToFloat();
        float intrp = in_value * slope + offset;
        data_ptr[i] = intrp;
      } else if (calc_mode_ == LutCalcMode::NF16) {
        NF16 in_quant = NF16::FromFloat(in_value);
        NF16 intrp = NF16::Add(NF16::Mul(in_quant, entry.slope), entry.offset);
        data_ptr[i] = intrp.ToFloat();
      } else if (calc_mode_ == LutCalcMode::NF16_FMA) {
        NF16 in_quant = NF16::FromFloat(in_value);
        NF16 intrp = NF16::Fma(in_quant, entry.slope, entry.offset);
        data_ptr[i] = intrp.ToFloat();
      } else {
        AIXV_CHECK(false);
      }
#endif
    }  // i
  }

 private:
  struct LutEntry {
    NF16 slope;
    NF16 offset;
    int lut1_sofs;
    int lut1_segs;
  };

  enum class LutCalcMode {
    FP32 = 0,
    NF16 = 1,
    NF16_FMA = 2,
  };

  std::string cfg_fname_;
  bool initialized_ = false;

  float range_lb_;
  float range_ub_;
  int lut0_size_;
  int lut1_size_;

  std::vector<LutEntry> lut0_entries_;
  std::vector<LutEntry> lut1_entries_;

  LutCalcMode calc_mode_;
};
}  // namespace

static bool kRegistered =
    Factory<X330Operation>::RegisterCreateFunction("Mish", MishLUT::Create);

std::unique_ptr<X330Operation> MishLUT::Create() {
  return std::make_unique<MishLUT>();
}

void MishLUT::PrepareQuantOperation(std::unique_ptr<Network>& network,
                                    const int idx_layer) {}

void MishLUT::Forward(Layer& layer, InferenceContext& ctx) {
  auto input = ctx.InputTensor(0);

  auto output = std::make_shared<Tensor>(input->n(), input->c(), input->h(),
                                         input->w(), input->dtype());
  std::memcpy(output->data(), input->data(), input->size());
  auto* out_data = output->data<float>();
  const auto out_dim_size = output->dimension().size();

  auto lut_model = AixLutModel("./configs/quantization/x330/mish.lut");
  lut_model.Activate(out_data, out_dim_size);

  ctx.SetOutputTensor(output);
}
}  // namespace x330
