#ifndef CPU_ACTS_ACTIVATION_HPP
#define CPU_ACTS_ACTIVATION_HPP

#include <functional>
#include <memory>

#include "datatype.hpp"
#include "inference_context.hpp"
#include "network/layer.hpp"
#include "network/tensor.hpp"
#include "operations/cpu_operation.hpp"

class Activation : public CpuOperation {
 public:
  void Forward(Layer &layer, InferenceContext &ctx) override final {
    input_ = ctx.InputTensor(0);
    auto dtype = ctx.out_dtype();
    InitOutputTensor(dtype);

    const bool is_fp32_forward = input_->dtype() == dty::DataType::FP32 &&
                                 output_->dtype() == dty::DataType::FP32;
    const bool is_fp64_forward = input_->dtype() == dty::DataType::FP64 &&
                                 output_->dtype() == dty::DataType::FP64;

    if (is_fp32_forward || is_fp64_forward) {
      ActivationForward(layer);
    } else {
      ActivationQuantForward(layer);
    }

    ctx.SetOutputTensor(output_);
  }
  virtual ~Activation() {}

 protected:
  virtual void InitOutputTensor(dty::DataType dtype) {
    output_ = std::make_shared<Tensor>(input_->n(), input_->c(), input_->h(),
                                       input_->w(), dtype);
  }
  virtual void ActivationForward(Layer &layer) = 0;
  virtual void ActivationQuantForward(Layer &layer) = 0;
  template <typename IType, typename OType>
  void ActivationForward(std::function<OType(IType)> &&activation) {
    const IType *in_data = input_->data<IType>();
    OType *out_data = output_->data<OType>();

    int each_height_size = output_->w();
    int each_channel_size = output_->h() * each_height_size;
    int each_batch_size = output_->c() * each_channel_size;

#pragma omp parallel for simd schedule(static) default(shared) collapse(4)
    for (int n = 0; n < output_->n(); ++n)
      for (int c = 0; c < output_->c(); ++c)
        for (int h = 0; h < output_->h(); ++h)
          for (int w = 0; w < output_->w(); ++w) {
            int idx = n * each_batch_size + c * each_channel_size +
                      h * each_height_size + w;

            out_data[idx] = activation(in_data[idx]);
          }
  }

  std::shared_ptr<Tensor> input_;
  std::shared_ptr<Tensor> output_;
};

#endif  // CPU_ACTS_ACTIVATION_HPP
