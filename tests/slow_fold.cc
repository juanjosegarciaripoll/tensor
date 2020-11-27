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
Tensor<typename Binop<n1, n2>::type> fold_11_11(const Tensor<n1> &A,
                                                const Tensor<n2> &B) {
  typedef typename Binop<n1, n2>::type n3;
  index a1, b1;
  A.get_dimensions(&a1);
  B.get_dimensions(&b1);
  assert(a1 == b1);

  Tensor<n3> output(1);
  n3 x = number_zero<n3>();
  for (index i = 0; i < a1; i++) {
    x += A(i) * B(i);
  }
  output.at(0) = x;
  return output;
}

template <typename n1, typename n2>
Tensor<typename Binop<n1, n2>::type> fold_12_11(const Tensor<n1> &A,
                                                const Tensor<n2> &B) {
  typedef typename Binop<n1, n2>::type n3;
  index a1, b1, b2;
  A.get_dimensions(&a1);
  B.get_dimensions(&b1, &b2);
  assert(a1 == b1);

  Tensor<n3> output(b2);
  for (index j = 0; j < b2; j++) {
    n3 x = number_zero<n3>();
    for (index i = 0; i < a1; i++) {
      x += A(i) * B(i, j);
    }
    output.at(j) = x;
  }
  return output;
}

template <typename n1, typename n2>
Tensor<typename Binop<n1, n2>::type> fold_13_11(const Tensor<n1> &A,
                                                const Tensor<n2> &B) {
  typedef typename Binop<n1, n2>::type n3;
  index a1, b1, b2, b3;
  A.get_dimensions(&a1);
  B.get_dimensions(&b1, &b2, &b3);
  assert(a1 == b1);

  Tensor<n3> output(b2, b3);
  for (index k = 0; k < b3; k++) {
    for (index j = 0; j < b2; j++) {
      n3 x = number_zero<n3>();
      for (index i = 0; i < a1; i++) {
        x += A(i) * B(i, j, k);
      }
      output.at(j, k) = x;
    }
  }
  return output;
}

template <typename n1, typename n2>
Tensor<typename Binop<n1, n2>::type> fold_21_11(const Tensor<n1> &A,
                                                const Tensor<n2> &B) {
  typedef typename Binop<n1, n2>::type n3;
  index a1, a2, b1;
  A.get_dimensions(&a1, &a2);
  B.get_dimensions(&b1);
  assert(a1 == b1);

  Tensor<n3> output(a2);

  for (index j = 0; j < a2; j++) {
    n3 x = number_zero<n3>();
    for (index i = 0; i < a1; i++) {
      x += A(i, j) * B(i);
    }
    output.at(j) = x;
  }
  return output;
}

template <typename n1, typename n2>
Tensor<typename Binop<n1, n2>::type> fold_22_11(const Tensor<n1> &A,
                                                const Tensor<n2> &B) {
  typedef typename Binop<n1, n2>::type n3;
  index a1, a2, b1, b2;
  A.get_dimensions(&a1, &a2);
  B.get_dimensions(&b1, &b2);
  assert(a1 == b1);

  Tensor<n3> output(a2, b2);

  for (index k = 0; k < b2; k++) {
    for (index j = 0; j < a2; j++) {
      n3 x = number_zero<n3>();
      for (index i = 0; i < a1; i++) {
        x += A(i, j) * B(i, k);
      }
      output.at(j, k) = x;
    }
  }
  return output;
}

template <typename n1, typename n2>
Tensor<typename Binop<n1, n2>::type> fold_23_11(const Tensor<n1> &A,
                                                const Tensor<n2> &B) {
  typedef typename Binop<n1, n2>::type n3;
  index a1, a2, b1, b2, b3;
  A.get_dimensions(&a1, &a2);
  B.get_dimensions(&b1, &b2, &b3);
  assert(a1 == b1);

  Tensor<n3> output(a2, b2, b3);

  for (index l = 0; l < b3; l++) {
    for (index k = 0; k < b2; k++) {
      for (index j = 0; j < a2; j++) {
        n3 x = number_zero<n3>();
        for (index i = 0; i < a1; i++) {
          x += A(i, j) * B(i, k, l);
        }
        output.at(j, k, l) = x;
      }
    }
  }
  return output;
}

template <typename n1, typename n2>
Tensor<typename Binop<n1, n2>::type> fold_31_11(const Tensor<n1> &A,
                                                const Tensor<n2> &B) {
  typedef typename Binop<n1, n2>::type n3;
  index a1, a2, a3, b1;
  A.get_dimensions(&a1, &a2, &a3);
  B.get_dimensions(&b1);
  assert(a1 == b1);

  Tensor<n3> output(a2, a3);

  for (index k = 0; k < a3; k++) {
    for (index j = 0; j < a2; j++) {
      n3 x = number_zero<n3>();
      for (index i = 0; i < a1; i++) {
        x += A(i, j, k) * B(i);
      }
      output.at(j, k) = x;
    }
  }
  return output;
}

template <typename n1, typename n2>
Tensor<typename Binop<n1, n2>::type> fold_32_11(const Tensor<n1> &A,
                                                const Tensor<n2> &B) {
  typedef typename Binop<n1, n2>::type n3;
  index a1, a2, a3, b1, b2;
  A.get_dimensions(&a1, &a2, &a3);
  B.get_dimensions(&b1, &b2);
  assert(a1 == b1);

  Tensor<n3> output(a2, a3, b2);

  for (index l = 0; l < b2; l++) {
    for (index k = 0; k < a3; k++) {
      for (index j = 0; j < a2; j++) {
        n3 x = number_zero<n3>();
        for (index i = 0; i < a1; i++) {
          x += A(i, j, k) * B(i, l);
        }
        output.at(j, k, l) = x;
      }
    }
  }
  return output;
}

template <typename n1, typename n2>
Tensor<typename Binop<n1, n2>::type> fold_33_11(const Tensor<n1> &A,
                                                const Tensor<n2> &B) {
  typedef typename Binop<n1, n2>::type n3;
  index a1, a2, a3, b1, b2, b3;
  A.get_dimensions(&a1, &a2, &a3);
  B.get_dimensions(&b1, &b2, &b3);
  assert(a1 == b1);

  Tensor<n3> output(a2, a3, b2, b3);

  for (index m = 0; m < b3; m++) {
    for (index l = 0; l < b2; l++) {
      for (index k = 0; k < a3; k++) {
        for (index j = 0; j < a2; j++) {
          n3 x = number_zero<n3>();
          for (index i = 0; i < a1; i++) {
            x += A(i, j, k) * B(i, l, m);
          }
          output.at(j, k, l, m) = x;
        }
      }
    }
  }
  return output;
}

template <typename n1, typename n2>
Tensor<typename Binop<n1, n2>::type> slow_fold(Tensor<n1> A, int ndx1,
                                               Tensor<n2> B, int ndx2) {
  if (ndx1 > A.rank()) {
    ndx1 = A.rank() + ndx1;
  }
  assert(ndx1 >= 0);
  while (ndx1 > 0) {
    A = permute(A, ndx1 - 1, ndx1);
    ndx1--;
  }
  if (ndx2 > B.rank()) {
    ndx2 = B.rank() + ndx2;
  }
  assert(ndx2 >= 0);
  while (ndx2 > 0) {
    B = permute(B, ndx2 - 1, ndx2);
    ndx2--;
  }
  switch (A.rank() * 10 + B.rank()) {
    case 11:
      return fold_11_11(A, B);
    case 12:
      return fold_12_11(A, B);
    case 13:
      return fold_13_11(A, B);
    case 21:
      return fold_21_11(A, B);
    case 22:
      return fold_22_11(A, B);
    case 23:
      return fold_23_11(A, B);
    case 31:
      return fold_31_11(A, B);
    case 32:
      return fold_32_11(A, B);
    case 33:
      return fold_33_11(A, B);
      // Unsupported case;
    default:
      return Tensor<typename Binop<n1, n2>::type>();
  }
}

}  // namespace tensor_test
