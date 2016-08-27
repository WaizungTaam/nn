#ifndef NN_LAYER_LAYER_H_
#define NN_LAYER_LAYER_H_

#include "../node.h"
#include <memory>


namespace nn {
  
using RealType = double;

namespace layer {

class Layer {
public:
  void link(const Layer& prev, const Layer& next) {
    node_ptr_->link_fwd(prev.node_ptr_, next.node_ptr_);
    node_ptr_->link_bwd(next.node_ptr_, prev.node_ptr_);
  }  
  virtual void forward() { node_ptr_->run_fwd(); }
  virtual void backward() { node_ptr_->run_bwd(); }
  virtual void update() { node_ptr_->update(); }

protected:
  void link_head(const Layer& next) {
    node_ptr_->link_fwd(NodePtr(nullptr), next.node_ptr_);
    node_ptr_->link_bwd(next.node_ptr_, NodePtr(nullptr));
  }
  void link_tail(const Layer& prev) {
    node_ptr_->link_fwd(prev.node_ptr_, NodePtr(nullptr));
    node_ptr_->link_bwd(NodePtr(nullptr), prev.node_ptr_);
  }

  NodePtr node_ptr_;
};

}  // namespace layer
}  // namespace nn

#endif  // NN_LAYER_LAYER_H_