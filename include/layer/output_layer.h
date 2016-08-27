#ifndef NN_LAYER_OUTPUT_LAYER_H_
#define NN_LAYER_OUTPUT_LAYER_H_

#include "layer.h"
#include "../node.h"
#include "../data.h"
#include "../loss.h"
#include "../tensor/matrix.h"

#include <memory>
#include "../tensor/util.h"  // DEBUG
#include <iostream>  // DEBUG


namespace nn {
namespace layer {

template <typename LossTp>
class OutputNode : public Node {
public:
  OutputNode(const tensor::Matrix<RealType>& data_out) :
    target_(data_out) {}
  tensor::Matrix<RealType> output() const {
    MatrixData<RealType> out = *std::dynamic_pointer_cast<
      MatrixData<RealType>>(fwd_prv_data_ptr_);
    return out();
  }

protected:
  void compute_fwd_() override {}
  void compute_bwd_() override {
    MatrixData<RealType> pred = *std::dynamic_pointer_cast<
      MatrixData<RealType>>(fwd_prv_data_ptr_);    
    bwd_nxt_data_cache_ = LossTp::df(pred(), target_);
    bwd_nxt_data_ptr_ = std::make_shared<MatrixData<RealType>>(
      bwd_nxt_data_cache_);
    // std::cout << "OutputLayer - backward:\n" << bwd_nxt_data_cache_() << "\n";  // DEBUG
    std::cout << tensor::util::square(bwd_nxt_data_cache_()).sum() /
                 (bwd_nxt_data_cache_().shape()[0] *
                  bwd_nxt_data_cache_().shape()[1]) << "\n"; // DEBUG
  }

  MatrixData<RealType> bwd_nxt_data_cache_;
  tensor::Matrix<RealType> target_;
};


template <typename LossTp>
class OutputLayer : public Layer {
public:
  OutputLayer() {}
  OutputLayer(const tensor::Matrix<RealType>& data_out) {
    node_ptr_ = std::make_shared<OutputNode<LossTp>>(
      OutputNode<LossTp>(data_out));
  }
  void link(const Layer& prev) {
    link_tail(prev);
  }
  void forward() override {
    std::dynamic_pointer_cast<OutputNode<LossTp>>(node_ptr_)->run_fwd();
  }
  void backward() override {
    std::dynamic_pointer_cast<OutputNode<LossTp>>(node_ptr_)->run_bwd(); 
  }
  void update() override {}
  tensor::Matrix<RealType> output() const {
    return std::dynamic_pointer_cast<OutputNode<LossTp>>(node_ptr_)->output();
  }  
};

}  // namespace layer
}  // namespace nn


#endif  // NN_LAYER_OUTPUT_LAYER_H_