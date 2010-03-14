// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
    Copyright (c) 2010 Juan Jose Garcia Ripoll

    Tensor is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public License as published
    by the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Library General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

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
