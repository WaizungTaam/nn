#include "../include/data/mnist.h"

#include <iostream>
#include <vector>

#include "../include/tensor/vector.h"
#include "../include/tensor/matrix.h"

#include "../include/model/mlp.h"
#include "../include/optimizer.h"
#include "../include/activ.h"
#include "../include/loss.h"


int main() {
  nn::data::MNIST mnist("data/MNIST/");
  tensor::Matrix<double> img_train, lbl_train, img_test, lbl_test;
  mnist.load_train(img_train, lbl_train);
  mnist.load_test(img_test, lbl_test);
  std::cout << "Data Loaded.\n";

  nn::model::MLP<nn::optimizer::Adam, nn::activ::Sigmoid, nn::loss::Squared> 
    model({784, 10, 10}, {4e-2, 0.4, 0.4, 1e-8});
  model.train(img_train, lbl_train, 10, 2000);
  tensor::Matrix<double> raw_pred = model.predict(img_test);
  tensor::Matrix<double> pred(raw_pred.shape()[0], raw_pred.shape()[1]);

  for (std::size_t i = 0; i < pred.shape()[0]; ++i) {
    pred[i] = (raw_pred[i] == raw_pred[i].max());
  }
  std::size_t num_correct = 0;
  for (std::size_t i = 0; i < pred.shape()[0]; ++i) {
    if (pred[i].equal(lbl_test[i])) ++num_correct;
  }
  std::cout << "Accuracy: " 
            << static_cast<double>(num_correct) / lbl_test.shape()[0]
            << "\n";
}

/*
Accuracy: 0.8951 => Adam, Sigmoid, Squared, {784, 100, 10}, {1e-2, 0.5, 0.5, 1e-8}, 10, 2000
Accuracy: 0.8956 => Adam, Sigmoid, Squared, {784, 100, 10}, {2e-2, 0.5, 0.5, 1e-8}, 10, 200
Accuracy: 0.8905 => Adam, Sigmoid, Squared, {784, 100, 10}, {2e-2, 0.5, 0.5, 1e-8}, 10, 6000
Accuracy: 0.9151 => Adam, Sigmoid, Squared, {784, 100, 10}, {2e-2, 0.5, 0.5, 1e-8}, 20, 6000
Accuracy: 0.9307 => Adam, Sigmoid, Squared, {784, 100, 10}, {2e-2, 0.5, 0.5, 1e-8}, 40, 6000
*/