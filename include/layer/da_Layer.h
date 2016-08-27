#ifndef NN_LAYER_DA_LAYER_H_
#define NN_LAYER_DA_LAYER_H_

#include "layer.h"
#include "../node.h"
#include "../data.h"
#include "../activ.h"
#include "../tensor/matrix.h"

#include <memory>


namespace nn {
namespace layer {

template <typename ActivTp>
class DANode : public Node {
public:
  DANode() {}
  DANode(const tensor::Matrix<RealType>& w,
         const tensor::Matrix<RealType>& wb_v,
         const tensor::Matrix<RealType>& wb_h) :
    weight_(w), w_bias_vis_(wb_v), w_bias_hid_(wb_h) {}
  void pretrain() {

  }
  void update() override {

  }
protected:
  void compute_fwd_() override {

  }
  void compute_bwd_() override {
    
  }

  MatrixData<RealType> fwd_nxt_data_cache_;
  MatrixData<RealType> bwd_nxt_data_cache_;

  tensor::Matrix<RealType> weight_;
  tensor::Matrix<RealType> w_bias_vis_;
  tensor::Matrix<RealType> w_bias_hid_;
};


template <typename ActivTp>
class DALayer : public Layer {
public:
  DALayer() {}
  DALayer(std::size_t dim_in, std::size_t dim_out)
};

}  // namespace layer
}  // namespace nn

#endif  // NN_LAYER_DA_LAYER_H_