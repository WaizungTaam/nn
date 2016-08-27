#include "../include/model/mlp.h"
#include "../include/tensor/matrix.h"
#include "../include/optimizer.h"
#include "../include/activ.h"
#include "../include/loss.h"


int main() {
  nn::model::MLP<nn::optimizer::Adam, nn::activ::Sigmoid, nn::loss::Squared> 
    model({2, 4, 1}, {4e-2, 0.9, 0.9, 1e-8});
  tensor::Matrix<double> x = {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
                         y = {{0}, {1}, {1}, {0}};
  model.train(x, y, 400, 2);
  std::cout << model.predict(x) << "\n";
}