// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include <cmath>
#include <tensor/tensor.h>
#include "loops.h"

namespace tensor_test {

  template<>
  Tensor<double> random_unitary(int n, int iterations)
  {
    Tensor<double> id = Tensor<double>::eye(n,n);
    if (n == 1)
      return id;
    Tensor<double> output = id;
    if (iterations <= 0)
      iterations = 2*n;
    while (iterations--) {
      int i = rand<int>(0, n), j;
      do {
        j = rand<int>(0, n);
      } while (i == j);
      Tensor<double> U = id;
      double theta = rand(0.0, M_PI);
      double c = cos(theta);
      double s = sin(theta);
      U.at(i,i) = c;
      U.at(j,j) = - c;
      U.at(i,j) = s;
      U.at(j,i) = s;
      output = mmult(U, output);
    }
    return output;
  }

} // namespace tensor_test
