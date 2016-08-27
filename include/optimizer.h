#ifndef NN_OPTIMIZER_H_
#define NN_OPTIMIZER_H_

#include "tensor/matrix.h"
#include "tensor/util.h"
#include <iostream> // DEBUG


namespace nn {
namespace optimizer {

using RealType = double;
using RealMatrix = tensor::Matrix<RealType>;

class Sgd {
public:
  typedef struct { 
    RealType learning_rate; RealType decay; RealType momentum; 
  } param_type;
  Sgd() {}
  Sgd(const RealType& lr, const RealType& dc, const RealType& mt) :
    learning_rate_(lr), decay_(dc), momentum_(mt) {}
  Sgd(const std::initializer_list<RealType>& il) :
    learning_rate_(*il.begin()), decay_(*(il.begin() + 1)),
    momentum_(*(il.begin() + 2)) {}
  Sgd(const param_type& param) :
    learning_rate_(param.learning_rate),
    decay_(param.decay),
    momentum_(param.momentum) {}
  param_type param() const {
    return param_type{learning_rate_, decay_, momentum_};
  }
  void update(RealMatrix& weight, RealMatrix& weight_cache, 
              RealMatrix& /*weight_cache_2*/, 
              const RealMatrix& gradient) const {
    weight_cache = momentum_ * weight_cache + learning_rate_ * (
                   gradient + decay_ * weight);
    weight = weight + weight_cache;
  }
private:
  RealType learning_rate_;
  RealType decay_;
  RealType momentum_;
};

class RMSprop {
public:
  typedef struct {
    RealType learning_rate; RealType decay; RealType epsilon;
  } param_type;
  RMSprop() {}
  RMSprop(const std::initializer_list<RealType>& il) :
    learning_rate_(*il.begin()), decay_(*(il.begin() + 1)),
    epsilon_(*(il.begin() + 2)) {}
  RMSprop(const RealType& lr, const RealType& dc, const RealType& ep) :
    learning_rate_(lr), decay_(dc), epsilon_(ep) {}
  RMSprop(const param_type& param) :
    learning_rate_(param.learning_rate), decay_(param.decay),
    epsilon_(param.epsilon) {}
  param_type param() const {
    return param_type{learning_rate_, decay_, epsilon_};
  }
  void update(RealMatrix& weight, RealMatrix& weight_cache, 
              RealMatrix& /*weight_cache_2*/, 
              const RealMatrix& gradient) const {
    weight_cache = decay_ * weight_cache + (1.0 - decay_) * 
                   tensor::util::square(gradient);
    weight = weight + learning_rate_ * gradient / tensor::util::sqrt(
             weight_cache + epsilon_);
  }
private:
  RealType learning_rate_;
  RealType decay_;
  RealType epsilon_;
};

class Adagrad {
public:
  typedef struct {
    RealType learning_rate; RealType epsilon;
  } param_type;
  Adagrad() {}
  Adagrad(const RealType& lr, const RealType& ep) :
    learning_rate_(lr), epsilon_(ep) {}
  Adagrad(const std::initializer_list<RealType>& il) :
    learning_rate_(*il.begin()), epsilon_(*(il.begin() + 1)) {}
  Adagrad(const param_type& param) :
    learning_rate_(param.learning_rate), epsilon_(param.epsilon) {}
  param_type param() const {
    return param_type{learning_rate_, epsilon_};
  }
  void update(RealMatrix& weight, RealMatrix& weight_cache, 
              RealMatrix& /*weight_cache_2*/, 
              const RealMatrix& gradient) const {
    weight_cache = weight_cache + tensor::util::square(gradient);
    weight = weight + learning_rate_ * gradient / tensor::util::sqrt(
             weight_cache + epsilon_);
  }
private:
  RealType learning_rate_;
  RealType epsilon_;
};

class Adam {
public:
  typedef struct {
    RealType learning_rate; RealType beta_1; RealType beta_2; RealType epsilon;
  } param_type;
  Adam() {}
  Adam(const RealType& lr, const RealType& b1, const RealType& b2,
       const RealType& ep) :
    learning_rate_(lr), beta_1_(b1), beta_2_(b2), epsilon_(ep) {}
  Adam(const std::initializer_list<RealType>& il) :
    learning_rate_(*il.begin()), beta_1_(*(il.begin() + 1)),
    beta_2_(*(il.begin() + 2)), epsilon_(*(il.begin() + 3)) {}
  Adam(const param_type& param) :
    learning_rate_(param.learning_rate),
    beta_1_(param.beta_1), beta_2_(param.beta_2), epsilon_(param.epsilon) {}
  param_type param() const {
    return param_type{learning_rate_, beta_1_, beta_2_, epsilon_};
  }
  void update(RealMatrix& weight, RealMatrix& weight_cache_1,
              RealMatrix& weight_cache_2, const RealMatrix& gradient) const {
    weight_cache_1 = beta_1_ * weight_cache_1 + (1.0 - beta_1_) * gradient;
    weight_cache_2 = beta_2_ * weight_cache_2 + (1.0 - beta_2_) *
                     tensor::util::square(gradient);                  
    weight = weight + learning_rate_ * weight_cache_1 / tensor::util::sqrt(
             weight_cache_2 + epsilon_);
  }
private:
  RealType learning_rate_;
  RealType beta_1_;
  RealType beta_2_;
  RealType epsilon_;
};

class Adadelta {
public:
  typedef struct {
    RealType decay; RealType rho; RealType epsilon;
  } param_type;
  Adadelta() {}
  Adadelta(const RealType& dc, const RealType& rh, const RealType& ep) :
    decay_(dc), rho_(rh), epsilon_(ep) {}
  Adadelta(const std::initializer_list<RealType>& il) :
    decay_(*il.begin()), rho_(*(il.begin() + 1)), 
    epsilon_(*(il.begin() + 2)) {}
  Adadelta(const param_type& param) :
    decay_(param.decay), rho_(param.rho), epsilon_(param.epsilon) {}
  param_type param() const {
    return param_type{decay_, rho_, epsilon_};
  }
  void update(RealMatrix& weight, RealMatrix& weight_cache_1,
              RealMatrix& weight_cache_2, const RealMatrix& gradient) const {
    weight_cache_1 = decay_ * weight_cache_1 + (1.0 - decay_) *
                     tensor::util::square(gradient);
    RealMatrix delta_weight = tensor::util::sqrt(
      (weight_cache_2 + epsilon_) / (weight_cache_1 + epsilon_)).times(
      gradient);
    weight_cache_2 = decay_ * weight_cache_2 + (1.0 - rho_) *
                     tensor::util::square(delta_weight);
    weight = weight + delta_weight;
  }
private:
  RealType decay_;
  RealType rho_;
  RealType epsilon_;
};

}  // optimizer
}  // namespace nn

#endif  // NN_OPTIMIZER_H_