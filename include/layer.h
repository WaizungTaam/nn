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

#ifndef NN_LAYER_H_
#define NN_LAYER_H_

#include "optimizer.h"
#include "activ.h"
#include "loss.h"
#include "tensor/matrix.h"

#include <iostream>


namespace nn {
namespace layer {

template <typename Data>
class Layer {
public:
  virtual Data forward(const Data& x) { return x; }
  virtual Data backward(const Data& x) { return x; }
  virtual void update() {}
};

using RealType = double;
using RealMatrix = tensor::Matrix<RealType>;

class InputLayer : public Layer<RealMatrix> {};

template <typename Optimizer>
class LinearLayer : public Layer<RealMatrix> {
public:
  LinearLayer(RealMatrix::size_type dim_in, RealMatrix::size_type dim_out,
              const std::initializer_list<RealType>& opt_param) : 
    weight_(
      RealMatrix(dim_in, dim_out, tensor::Random::uniform_real, -1.0, 1.0)),
    w_bias_(
      RealMatrix(1, dim_out, tensor::Random::uniform_real, -1.0, 1.0)),
    optimizer_(Optimizer(opt_param)),
    weight_cache_1_(RealMatrix(dim_in, dim_out)),
    weight_cache_2_(RealMatrix(dim_in, dim_out)),
    w_bias_cache_1_(RealMatrix(1, dim_out)),
    w_bias_cache_2_(RealMatrix(1, dim_out)),
    x_(RealMatrix()), g_(RealMatrix()) {}

  RealMatrix forward(const RealMatrix& x) override {
    x_ = x;
    RealMatrix bias(x.shape()[0], 1, 1.0);
    return x * weight_ + bias * w_bias_;
  }
  RealMatrix backward(const RealMatrix& g) override {
    g_ = g;
    return g * weight_.T();
  }
  void update() override {
    RealMatrix bias(x_.shape()[0], 1, 1.0);
    optimizer_.update(weight_, weight_cache_1_, weight_cache_2_, x_.T() * g_);
    optimizer_.update(w_bias_, w_bias_cache_1_, w_bias_cache_2_, 
      bias.T() * g_);
  }

private:
  RealMatrix weight_;
  RealMatrix w_bias_;

  Optimizer optimizer_;

  RealMatrix weight_cache_1_;
  RealMatrix weight_cache_2_;
  RealMatrix w_bias_cache_1_;
  RealMatrix w_bias_cache_2_;

  RealMatrix x_;
  RealMatrix g_;
};

template <typename Activ>
class ActivLayer : public Layer<RealMatrix> {
public:
  ActivLayer() : y_(RealMatrix()) {}

  RealMatrix forward(const RealMatrix& x) override {
    y_ = Activ::f(x);
    return y_;
  }
  RealMatrix backward(const RealMatrix& g) override {
    return g.times(Activ::df(y_));
  }

private:
  RealMatrix y_;
};

template <typename Loss>
class OutputLayer : public Layer<RealMatrix> {
public:
  OutputLayer() : p_(RealMatrix()) {}

  RealMatrix forward(const RealMatrix& p) override {
    p_ = p;
    return p;
  }
  RealMatrix backward(const RealMatrix& t) override {
    std::cout << Loss::f(p_, t).sum() << "\n";
    return Loss::df(p_, t);
  }

private:
  RealMatrix p_;
};

}  // namespace layer
}  // namespace nn

#endif  // NN_LAYER_H_