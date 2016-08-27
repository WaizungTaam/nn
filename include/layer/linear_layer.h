#ifndef NN_LAYER_LINEAR_LAYER_H_
#define NN_LAYER_LINEAR_LAYER_H_

#include "layer.h"
#include "../node.h"
#include "../data.h"
#include "../optimizer.h"
#include "../tensor/matrix.h"

#include <memory>
// #include <iostream>  // DEBUG


namespace nn {
namespace layer {

template <typename OptmizerTp>
class LinearNode : public Node {
public:
  LinearNode() {}
  LinearNode(const tensor::Matrix<RealType>& w,
             const tensor::Matrix<RealType>& wb,
             const OptmizerTp& opt) : 
    weight_(w), w_bias_(wb), optimizer_(opt),
    weight_cache_1(
      tensor::Matrix<RealType>(w.shape()[0], w.shape()[1], 0.0)),
    weight_cache_2(
      tensor::Matrix<RealType>(w.shape()[0], w.shape()[1], 0.0)),
    w_bias_cache_1(
      tensor::Matrix<RealType>(wb.shape()[0], wb.shape()[1], 0.0)),
    w_bias_cache_2(
      tensor::Matrix<RealType>(wb.shape()[0], wb.shape()[1], 0.0)) {}
  void update() override {
    MatrixData<RealType> x = *std::dynamic_pointer_cast<
      MatrixData<RealType>>(fwd_prv_data_ptr_);
    MatrixData<RealType> gradient = *std::dynamic_pointer_cast<
      MatrixData<RealType>>(bwd_prv_data_ptr_);      
    tensor::Matrix<RealType> bias(x().shape()[0], 1, 1.0);
    optimizer_.update(
      weight_, weight_cache_1, weight_cache_2, 
      x().T() * gradient());
    optimizer_.update(
      w_bias_, w_bias_cache_1, w_bias_cache_2, 
     bias.T() * gradient());
    // std::cout << "weight:\n" << weight_ << "\n"   // DEBUG
              // << "w_bias:\n" << w_bias_ << "\n";  // DEBUG
  }

protected:
  void compute_fwd_() override {
    MatrixData<RealType> x = *std::dynamic_pointer_cast<
      MatrixData<RealType>>(fwd_prv_data_ptr_);
    tensor::Matrix<RealType> bias(x().shape()[0], 1, 1.0);
    fwd_nxt_data_cache_ = x() * weight_ + bias * w_bias_;
    fwd_nxt_data_ptr_ = std::make_shared<MatrixData<RealType>>(
      fwd_nxt_data_cache_);
    // std::cout << "LinearLayer - forward:\n" << fwd_nxt_data_cache_() << "\n";  // DEBUG
  }
  void compute_bwd_() override {
    MatrixData<RealType> gradient = *std::dynamic_pointer_cast<
      MatrixData<RealType>>(bwd_prv_data_ptr_);
    bwd_nxt_data_cache_ = gradient() * weight_.T();
    bwd_nxt_data_ptr_ = std::make_shared<MatrixData<RealType>>(
      bwd_nxt_data_cache_);
    // std::cout << "LinearLayer - backward:\n" << bwd_nxt_data_cache_() << "\n";  // DEBUG
  }

  MatrixData<RealType> fwd_nxt_data_cache_;
  MatrixData<RealType> bwd_nxt_data_cache_;

  OptmizerTp optimizer_;

  tensor::Matrix<RealType> weight_;
  tensor::Matrix<RealType> w_bias_;

  tensor::Matrix<RealType> weight_cache_1;
  tensor::Matrix<RealType> weight_cache_2;
  tensor::Matrix<RealType> w_bias_cache_1;
  tensor::Matrix<RealType> w_bias_cache_2;
};


template <typename OptmizerTp>
class LinearLayer : public Layer {
public:
  LinearLayer() {}
  LinearLayer(std::size_t dim_in, std::size_t dim_out, const OptmizerTp& opt) {
    tensor::Matrix<RealType> weight(
      dim_in, dim_out, tensor::Random::uniform_real, -1.0, 1.0);
    tensor::Matrix<RealType> w_bias(
      1, dim_out, tensor::Random::uniform_real, -1.0, 1.0);
    node_ptr_ = std::make_shared<LinearNode<OptmizerTp>>(
      LinearNode<OptmizerTp>(weight, w_bias, opt));
  }
  void forward() override { 
    std::dynamic_pointer_cast<LinearNode<OptmizerTp>>(node_ptr_)->run_fwd(); 
  }
  void backward() override {
    std::dynamic_pointer_cast<LinearNode<OptmizerTp>>(node_ptr_)->run_bwd(); 
  }
  void update() override { 
    std::dynamic_pointer_cast<LinearNode<OptmizerTp>>(node_ptr_)->update(); 
  }
};

}  // namespace layer
}  // namespace nn

#endif  // NN_LAYER_LINEAR_LAYER_H_