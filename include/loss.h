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

#ifndef NN_LOSS_H_
#define NN_LOSS_H_

#include "tensor/matrix.h"
#include "tensor/util.h"


namespace nn {
namespace loss {

using RealType = double;
using RealMatrix = tensor::Matrix<RealType>;

struct Absolute {
  static RealMatrix f(const RealMatrix& prediction, 
                      const RealMatrix& target) {
    return tensor::util::abs(target - prediction);
  }
  static RealMatrix df(const RealMatrix& prediction, 
                       const RealMatrix& target) {
    return (target > prediction) - (target < prediction);
  }
};

struct Squared {
  static RealMatrix f(const RealMatrix& prediction, 
                      const RealMatrix& target) {
    return 0.5 * tensor::util::square(target - prediction);
  }
  static RealMatrix df(const RealMatrix& prediction, 
                       const RealMatrix& target) {
    return target - prediction;
  }
};

struct Hinge {
  // http://lasagne.readthedocs.io/en/latest/modules/objectives.html
  // https://en.wikipedia.org/wiki/Hinge_loss
  // https://github.com/Lasagne/Lasagne/blob/master/lasagne/objectives.py#L252-L288
  // https://github.com/JohnLangford/vowpal_wabbit/wiki/Loss-functions
  // http://image.diku.dk/shark/sphinx_pages/build/html/rest_sources/tutorials/concepts/library_design/losses.html
};

struct CrossEntropy {
  // https://en.wikipedia.org/wiki/Cross_entropy
  // http://neuralnetworksanddeeplearning.com/chap3.html#introducing_the_cross-entropy_cost_function
  // http://image.diku.dk/shark/doxygen_pages/html/classshark_1_1_cross_entropy.html

};


}  // namespace loss
}  // namespace nn


#endif  // NN_LOSS_H_