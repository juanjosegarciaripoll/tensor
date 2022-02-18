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

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <tensor/tensor.h>
#include "loops.h"

namespace tensor_test {

template <>
CTensor random_unitary(int n, int iterations) {
  CTensor id = CTensor::eye(n, n);
  if (n == 1) return id;
  CTensor output = id;
  if (iterations <= 0) iterations = 2 * n;
  while (iterations--) {
    int j, i = rand<int>(0, n);
    do {
      j = rand<int>(0, n);
    } while (i == j);
    CTensor U = id;
    double theta = rand(0.0, M_PI);
    double c = cos(theta);
    double s = sin(theta);
    double phase = rand(0.0, M_PI);
    cdouble ph = to_complex(s * cos(phase), s * sin(phase));
    U.at(i, i) = c;
    U.at(j, j) = -c;
    U.at(i, j) = ph;
    U.at(j, i) = tensor::conj(ph);
    output = mmult(U, output);
  }
  return output;
}

}  // namespace tensor_test
