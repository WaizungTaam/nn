#include "../include/sequential.h"
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

  nn::model::Sequential model(
    {&input, &linear_1, &activ_1, &linear_2, &activ_2, &output}, 1000, 2);

  model.train(x, y);
  std::cout << model.predict(x) << "\n";
}