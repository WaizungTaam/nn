#include "../include/layer/layer.h"
#include "../include/layer/input_layer.h"
#include "../include/layer/linear_layer.h"
#include "../include/layer/activ_layer.h"
#include "../include/layer/output_layer.h"

#include "../include/tensor/matrix.h"
#include "../include/activ.h"
#include "../include/loss.h"
#include "../include/optimizer.h"

#include <iostream>



int main() {
  tensor::Matrix<double> x = {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
                         y = {{0}, {1}, {1}, {0}};

  nn::InputLayer input(x);
  nn::LinearLayer<nn::optimizer::Sgd> linear_1(
    2, 4, nn::optimizer::Sgd(4e-2, 1e-3, 0.8));
  // nn::LinearLayer<nn::optimizer::Adam> linear_1(
  //   2, 4, nn::optimizer::Adam(8e-2, 8e-1, 9e-1, 1e-8));
  nn::ActivLayer<nn::activ::Sigmoid> activ_1;
  nn::LinearLayer<nn::optimizer::Sgd> linear_2(
    4, 1, nn::optimizer::Sgd(4e-2, 1e-3, 0.8));
  // nn::LinearLayer<nn::optimizer::Adam> linear_2(
  //   4, 1, nn::optimizer::Adam(8e-2, 8e-1, 9e-1, 1e-8));
  nn::ActivLayer<nn::activ::Sigmoid> activ_2;
  nn::OutputLayer<nn::loss::Squared> output(y);

  input.link(linear_1);
  linear_1.link(input, activ_1);
  activ_1.link(linear_1, linear_2);
  linear_2.link(activ_1, activ_2);
  activ_2.link(linear_2, output);
  output.link(activ_2);

  for (std::size_t i = 0; i < 10000; ++i) {
    std::cout << i << " ";

    input.forward();
    linear_1.forward();
    activ_1.forward();
    linear_2.forward();
    activ_2.forward();
    output.forward();

    output.backward();
    activ_2.backward();
    linear_2.backward();
    activ_1.backward();
    linear_1.backward();
    input.backward();

    linear_1.update();
    linear_2.update();

    std::cout << "\n";
  }
}