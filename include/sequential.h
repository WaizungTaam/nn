/* 
 * Copyright 2016 Waizung Taam
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NN_SEQUENTIAL_H_
#define NN_SEQUENTIAL_H_

#include "layer.h"
#include <iostream>
#include <functional>
#include <vector>


namespace nn {
namespace model {

using RealType = double;
using RealMatrix = tensor::Matrix<RealType>;

class Sequential {
public:
  Sequential(const std::initializer_list<layer::Layer<RealMatrix>*>& layers, 
             std::size_t num_epochs, std::size_t batch_size) :
    layers_(layers), num_epochs_(num_epochs), batch_size_(batch_size) {}

  void train(const RealMatrix& x, const RealMatrix& t) {
    mini_batch(x, t);
  }
  RealMatrix predict(const RealMatrix& x) {
    return forward(x);
  }

private:
  RealMatrix forward(const RealMatrix& x) {
    RealMatrix y = layers_[0]->forward(x);
    for (std::size_t i = 1; i < layers_.size(); ++i) {
      y = layers_[i]->forward(y);
    }
    return y;
  }
  void backward(const RealMatrix& t) {
    RealMatrix g = layers_[layers_.size() - 1]->backward(t);
    for (long i = layers_.size() - 2; i >= 0; --i) {
      g = layers_[i]->backward(g);
    }
  }
  void update() {
    for (std::size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->update();
    }
  }
  void train_once(const RealMatrix& x, const RealMatrix& t) {
    forward(x);
    backward(t);
    update();
  }
  std::vector<RealMatrix> shuffle(const RealMatrix& x, const RealMatrix& t) {
    RealMatrix data = x.insert(t, 1, x.shape()[1]);
    RealMatrix rand_data = data.shuffle();
    RealMatrix x_rand = rand_data(0, rand_data.shape()[0], 0, x.shape()[1]),
               t_rand = rand_data(
                0, rand_data.shape()[0], x.shape()[1], rand_data.shape()[1]);
    return std::vector<RealMatrix>{x_rand, t_rand};
  }
  void mini_batch(const RealMatrix& x, const RealMatrix& t) {
    std::size_t num_samples = x.shape()[0],
                num_batches = static_cast<std::size_t>(
      num_samples / batch_size_);
    if (num_batches * batch_size_ != num_samples) {
      num_batches = num_batches + 1;
    }
    for (std::size_t idx_epoch = 0; idx_epoch < num_epochs_; ++idx_epoch) {
      std::vector<RealMatrix> rand_data = shuffle(x,  t);
      RealMatrix x_rand = rand_data[0], t_rand = rand_data[1];
      for (std::size_t idx_batch = 0; idx_batch < num_batches; ++idx_batch) {
        std::size_t batch_begin = idx_batch * batch_size_,
                    batch_end = (idx_batch + 1) * batch_size_;
        if (idx_batch == num_batches - 1) {
          batch_end = num_samples;
        }
        std::cout << idx_epoch << "\t" << idx_batch << "\t";
        train_once(x_rand(batch_begin, batch_end), 
                   t_rand(batch_begin, batch_end));
      }
    }
  }

  std::size_t num_epochs_;
  std::size_t batch_size_;
  std::vector<layer::Layer<RealMatrix>*> layers_;
};

}  // namespace model
}  // namespace nn

#endif  // NN_SEQUENTIAL_H_