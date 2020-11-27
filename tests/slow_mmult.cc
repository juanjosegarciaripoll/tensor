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

#include <cassert>
#define TENSOR_LOAD_IMPL
#include <tensor/tensor.h>

namespace tensor_test {

using namespace tensor;
using tensor::index;

template <typename n1, typename n2>
Tensor<typename Binop<n1, n2>::type> fold_22_12(const Tensor<n1> &A,
                                                const Tensor<n2> &B) {
  typedef typename Binop<n1, n2>::type n3;
  index a1, a2, b1, b2;
  A.get_dimensions(&a1, &a2);
  B.get_dimensions(&b1, &b2);
  assert(a2 == b1);

  Tensor<n3> output(a1, b2);

  for (index i = 0; i < a1; i++) {
    for (index k = 0; k < b2; k++) {
      n3 x = number_zero<n3>();
      for (index j = 0; j < a2; j++) {
        x += A(i, j) * B(j, k);
      }
      output.at(i, k) = x;
    }
  }
  return output;
}

}  // namespace tensor_test
