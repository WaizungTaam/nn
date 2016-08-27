#ifndef NN_LAYER_NODE_H_
#define NN_LAYER_NODE_H_

#include "data.h"
#include <memory>


namespace nn {

class Node;
using NodePtr = std::shared_ptr<Node>;

class Node {
public:
  virtual ~Node() {}

  void link_fwd(NodePtr prv, NodePtr nxt) {
    fwd_prv_node_ptr_ = prv;
    fwd_nxt_node_ptr_ = nxt;
  }
  void link_bwd(NodePtr prv, NodePtr nxt) {
    bwd_prv_node_ptr_ = prv;
    bwd_nxt_node_ptr_ = nxt;
  }

  void run_fwd() {
    if (fwd_prv_node_ptr_) {
      fwd_prv_data_ptr_ = fwd_prv_node_ptr_->fwd_nxt_data_ptr_;
    }
    compute_fwd_();
  }
  void run_bwd() {
    if (bwd_prv_node_ptr_) {
      bwd_prv_data_ptr_ = bwd_prv_node_ptr_->bwd_nxt_data_ptr_;
    }
    compute_bwd_();
  }

  virtual void update() {}

protected:
  virtual void compute_fwd_() = 0;
  virtual void compute_bwd_() = 0;

  NodePtr fwd_prv_node_ptr_;
  NodePtr fwd_nxt_node_ptr_;
  DataPtr fwd_prv_data_ptr_;
  DataPtr fwd_nxt_data_ptr_;

  NodePtr bwd_prv_node_ptr_;
  NodePtr bwd_nxt_node_ptr_;
  DataPtr bwd_prv_data_ptr_;
  DataPtr bwd_nxt_data_ptr_;
};

}  // namespace nn

#endif  // NN_LAYER_NODE_H_