#ifndef NN_LAYER_ACTIV_LAYER_H_
#define NN_LAYER_ACTIV_LAYER_H_

#include "layer.h"
#include "../node.h"
#include "../data.h"
#include "../tensor/matrix.h"
#include "../activ.h"

#include <memory>
// #include <iostream>  // DEBUG


namespace nn {
namespace layer {

template <typename ActivTp>
class ActivNode : public Node {
public:
  ActivNode() {}

protected:
  void compute_fwd_() override {
    MatrixData<RealType> x = *std::dynamic_pointer_cast<MatrixData<RealType>>(
      fwd_prv_data_ptr_);
    fwd_nxt_data_cache_ = ActivTp::f(x());
    fwd_nxt_data_ptr_ = std::make_shared<MatrixData<RealType>>(
      fwd_nxt_data_cache_);
    // std::cout << "ActivLayer - forward:\n" << fwd_nxt_data_cache_() << "\n";  // DEBUG
  }
  void compute_bwd_() override {
    MatrixData<RealType> grad = *std::dynamic_pointer_cast<MatrixData<RealType>>(
      bwd_prv_data_ptr_);
    MatrixData<RealType> y = fwd_nxt_data_cache_;
    bwd_nxt_data_cache_ = grad().times(ActivTp::df(y()));
    bwd_nxt_data_ptr_ = std::make_shared<MatrixData<RealType>>(
      bwd_nxt_data_cache_);
    // std::cout << "ActivLayer - backward:\n" << bwd_nxt_data_cache_() << "\n";  // DEBUG
  }

  MatrixData<RealType> fwd_nxt_data_cache_;
  MatrixData<RealType> bwd_nxt_data_cache_;
};


template <typename ActivTp>
class ActivLayer : public Layer {
public:
  ActivLayer() {
    node_ptr_ = std::make_shared<ActivNode<ActivTp>>(ActivNode<ActivTp>());
  }
  void forward() override {
    std::dynamic_pointer_cast<ActivNode<ActivTp>>(node_ptr_)->run_fwd();
  }
  void backward() override {
    std::dynamic_pointer_cast<ActivNode<ActivTp>>(node_ptr_)->run_bwd();
  }
  void update() override {}
};

}  // namespace layer
}  // namespace nn


#endif  // NN_LAYER_ACTIV_LAYER_H_