#ifndef NN_ACTIV_H_
#define NN_ACTIV_H_

#include "tensor/matrix.h"
#include "tensor/util.h"


namespace nn {
namespace activ {

using RealType = double;
using RealMatrix = tensor::Matrix<RealType>;


struct Identity {
  static RealMatrix f(const RealMatrix& x) { return x; }
  static RealMatrix df(const RealMatrix& x) {
    return RealMatrix(x.shape()[0], x.shape()[1], 1.0);
  }
};

struct ReLU {
  static RealMatrix f(const RealMatrix& x) { 
    return tensor::util::relu(x); 
  }
  static RealMatrix df(const RealMatrix& x) { 
    return tensor::util::d_relu(x);
  }
};

struct LeakyReLu {
  static RealMatrix f(const RealMatrix& x) {
    return tensor::util::leaky_relu(x);
  }
  static RealMatrix df(const RealMatrix& x) {
    return tensor::util::d_leaky_relu(x);
  }
};

struct ELU {
  static RealMatrix f(const RealMatrix& x) {
    return tensor::util::elu(x);
  }
  static RealMatrix df(const RealMatrix& x) {
    return tensor::util::d_elu(x);
  }
};

struct Sigmoid {
  static RealMatrix f(const RealMatrix& x) {
    return tensor::util::sigmoid(x);
  }
  static RealMatrix df(const RealMatrix& x) {
    return tensor::util::d_sigmoid(x);
  }
};

struct Tanh {
  static RealMatrix f(const RealMatrix& x) {
    return tensor::util::tanh(x);
  }
  static RealMatrix df(const RealMatrix& x) {
    return tensor::util::d_tanh(x);
  }
};

struct Softmax {
  static RealMatrix f(const RealMatrix& x) {
    return tensor::util::softmax(x);
  }
  static RealMatrix df(const RealMatrix& x) {
    return tensor::util::d_softmax(x);
  }
};

struct Softplus {
  static RealMatrix f(const RealMatrix& x) {
    return tensor::util::softplus(x);
  }
  static RealMatrix df(const RealMatrix& x) {
    return tensor::util::d_softplus(x);
  }
};


}  // namespace activ
}  // namespace nn

#endif  // NN_ACTIV_H_