#ifndef NN_MODEL_MLP_H_
#define NN_MODEL_MLP_H_

#include "../layer/input_layer.h"
#include "../layer/linear_layer.h"
#include "../layer/activ_layer.h"
#include "../layer/output_layer.h"

#include "../optimizer.h"
#include "../activ.h"
#include "../loss.h"
#include "../tensor/matrix.h"

#include <type_traits>
#include <vector>
// #include <iostream>  // DEBUG


namespace nn {
namespace model {

template <typename OptimizerTp, typename ActivTp, typename LossTp>
class MLP {
public:
  MLP(const std::initializer_list<std::size_t>& layer_sizes,
      const std::initializer_list<RealType>& opt_param) {
    // std::cout << "In MLP()\n"; // DEBUG
    linears_.resize(layer_sizes.size() - 1);
    activs_.resize(layer_sizes.size() - 1);
    std::vector<std::size_t> vec_layer_sizes(layer_sizes);
    for (std::size_t idx = 0; idx < vec_layer_sizes.size() - 1; ++idx) {
      linears_[idx] = layer::LinearLayer<OptimizerTp>(
        vec_layer_sizes[idx], vec_layer_sizes[idx + 1], 
        OptimizerTp(opt_param));
      activs_[idx] = layer::ActivLayer<ActivTp>();
    }
    // std::cout << "MLP() OK\n"; // DEBUG
  }
  void train(const tensor::Matrix<RealType>& x, 
             const tensor::Matrix<RealType>& y,
             std::size_t num_epochs, std::size_t batch_size) {
    // std::cout << "In train()\n"; // DEBUG
    std::size_t num_batches = static_cast<std::size_t>(
      x.shape()[0] / batch_size);
    if (num_batches * batch_size != x.shape()[0]) {
      num_batches = num_batches + 1;
    }
    tensor::Matrix<RealType> data = x.insert(y, 1, x.shape()[1]);
    tensor::Matrix<RealType> data_rand = data.shuffle();
    tensor::Matrix<RealType> x_rand = data_rand(
      0, data_rand.shape()[0], 0, x.shape()[1]);
    tensor::Matrix<RealType> y_rand = data_rand(
      0, data_rand.shape()[0], x.shape()[1], data_rand.shape()[1]);
    for (std::size_t idx_epoch = 0; idx_epoch < num_epochs; ++idx_epoch) {
      for (std::size_t idx_batch = 0; idx_batch < num_batches; ++idx_batch) {
        std::size_t idx_batch_begin = idx_batch * batch_size;
        std::size_t idx_batch_end = (idx_batch + 1) * batch_size;
        if (idx_batch == num_batches - 1) {
          idx_batch_end = x_rand.shape()[0];
        }
        input_ = layer::InputLayer(x_rand(idx_batch_begin, idx_batch_end));
        output_ = layer::OutputLayer<LossTp>(
          y_rand(idx_batch_begin, idx_batch_end));
        link_all_layers_();
        std::cout << idx_epoch << "\t" << idx_batch << "\t"; // DEBUG
        forward_();
        backward_();
        update_();
      }
    }
    // std::cout << "train() OK\n"; // DEBUG
  }
  tensor::Matrix<RealType> predict(const tensor::Matrix<RealType>& x) {
    input_ = layer::InputLayer(x);
    link_all_layers_();
    forward_();
    return output_.output();
  }

private:
  void link_all_layers_() {
    // std::cout << "In link_all_layers_()\n"; // DEBUG
    std::size_t num_layers = linears_.size();
    input_.link(linears_[0]);
    linears_[0].link(input_, activs_[0]);
    activs_[0].link(linears_[0], linears_[1]);
    for (std::size_t idx = 1; idx < num_layers - 1; ++idx) {
      linears_[idx].link(activs_[idx - 1], activs_[idx]);
      activs_[idx].link(linears_[idx], linears_[idx + 1]);
    }
    linears_[num_layers - 1].link(
      activs_[num_layers - 2], activs_[num_layers - 1]);
    activs_[num_layers - 1].link(linears_[num_layers - 1], output_);
    output_.link(activs_[num_layers - 1]);
    // std::cout << "link_all_layers_() OK\n"; // DEBUG
  }
  void forward_() {
    // std::cout << "In forward_()\n"; // DEBUG
    input_.forward();
    for (std::size_t idx = 0; idx < linears_.size(); ++idx) {
      linears_[idx].forward();
      activs_[idx].forward();
    }
    output_.forward();
    // std::cout << "forward_ OK\n"; // DEBUG
  }
  void backward_() {
    // std::cout << "In backward_\n"; // DEBUG
    output_.backward();
    // std::cout << "output backward OK\n"; // DEBUG
    for (std::make_signed<std::size_t>::type idx = linears_.size() - 1; 
         idx >= 0; --idx) {
      // std::cout << "idx: " << idx << "\n"; // DEBUG
      activs_[idx].backward();
      linears_[idx].backward();
    }
    // std::cout << "Linear and Activ backward OK\n";
    input_.backward();
    // std::cout << "backward_ OK\n"; // DEBUG
  }
  void update_() {
    // std::cout << "In update_\n"; // DEBUG
    for (std::size_t idx = 0; idx < linears_.size(); ++idx) {
      linears_[idx].update();
    }
    // std::cout << "update_ OK\n"; // DEBUG
  }

  layer::InputLayer input_;
  std::vector<layer::LinearLayer<OptimizerTp>> linears_;
  std::vector<layer::ActivLayer<ActivTp>> activs_;
  layer::OutputLayer<LossTp> output_;
};

}  // namespace model
}  // namespace nn

#endif  // NN_MODEL_MLP_H_