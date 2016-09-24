#include "../include/layer.h"

#include "../include/optimizer.h"
#include "../include/activ.h"
#include "../include/loss.h"
#include "../include/tensor/matrix.h"

#include <iostream>

int main() {
  tensor::Matrix<double> x = {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
                         y = {{0}, {1}, {1}, {0}};
  nn::layer::InputLayer input;
  nn::layer::LinearLayer<nn::optimizer::Sgd> linear_1(2, 8, {1e-1, 1e-3, 8e-1});
  nn::layer::ActivLayer<nn::activ::Sigmoid> activ_1;
  nn::layer::LinearLayer<nn::optimizer::Sgd> linear_2(8, 1, {1e-1, 1e-3, 8e-1});
  nn::layer::ActivLayer<nn::activ::Sigmoid> activ_2;
  nn::layer::OutputLayer<nn::loss::Squared> output;

  for (std::size_t i = 0; i < 1000; ++i) {
    auto v = input.forward(x);
    v = linear_1.forward(v);
    v = activ_1.forward(v);
    v = linear_2.forward(v);
    v = activ_2.forward(v);
    v = output.forward(v);

    auto g = output.backward(y);
    g = activ_2.backward(g);
    g = linear_2.backward(g);
    g = activ_1.backward(g);
    g = linear_1.backward(g);
    g = input.backward(g);

    input.update();
    linear_1.update();
    activ_1.update();
    linear_2.update();
    activ_2.update();
    output.update();
  }  
}