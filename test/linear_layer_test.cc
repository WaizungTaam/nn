#include "../include/layer/linear_layer.h"

int main() {
  nn::LinearLayer a(3, 4), b(4, 5), c(5, 6);
  b.link(a, c);
}