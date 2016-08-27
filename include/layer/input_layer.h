#ifndef NN_LAYER_INPUT_LAYER_H_
#define NN_LAYER_INPUT_LAYER_H_

#include "layer.h"
#include "../node.h"
#include "../data.h"
#include "../tensor/matrix.h"

#include <memory>
// #include <iostream>  // DEBUG


namespace nn {
namespace layer {

class InputNode : public Node {
public:
  InputNode(const tensor::Matrix<RealType>& data_in) : 
    fwd_nxt_data_cache_(data_in) {}

protected:
  void compute_fwd_() override {
    fwd_nxt_data_ptr_ = std::make_shared<MatrixData<RealType>>(
      fwd_nxt_data_cache_);
    // std::cout << "InputLayer - forward:\n" << fwd_nxt_data_cache_() << "\n";  // DEBUG
  }
  void compute_bwd_() override {}

  MatrixData<RealType> fwd_nxt_data_cache_;
};


class InputLayer : public Layer {
public:
  InputLayer() {}
  InputLayer(const tensor::Matrix<RealType>& data_in) {
    node_ptr_ = std::make_shared<InputNode>(InputNode(data_in));
  }
  void link(const Layer& next) {
    link_head(next);
  }
  void forward() override {
    std::dynamic_pointer_cast<InputNode>(node_ptr_)->run_fwd();
  }
  void backward() override {
    std::dynamic_pointer_cast<InputNode>(node_ptr_)->run_bwd(); 
  }
  void update() override {}
};

}  // namespace layer
}  // namespace nn


#endif  // NN_LAYER_INPUT_LAYER_H_