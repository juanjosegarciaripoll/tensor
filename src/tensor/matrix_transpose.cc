// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include <cassert>
#include <tensor/tensor.h>

namespace tensor {

  template<typename n> inline
  const Tensor<n> do_transpose(const Tensor<n> &a)
  {
    assert(a.rank() == 2);
    index rows = a.rows();
    index cols = a.columns();
    Tensor<n> b(cols, rows);
    //
    // Matrix A is in row major order, with elements contiguously
    // as follows: (0,0), (1,0), (2,0), ... (i,j) ... (rows-1,cols-1)
    // Matrix B is the transpose, so that
    // (0,0), (0,1), (0,2), ... (1,0), (1,1), ... (j,i)...
    //
    typename Tensor<n>::const_iterator ij_a = a.begin();
    typename Tensor<n>::iterator j_b = b.begin();
    // B(j,i) = A(i,j)
    // i = 0 .. rows-1;
    // j = 0 .. cols-1;
    for (int j = cols; j; j--, j_b++) {
      typename Tensor<n>::iterator ij_b = j_b;
      for (int i = rows; i; i--, ij_a++, ij_b += cols) {
        *ij_b = *ij_a;
      }
    }
    return b;
  }

} // namespace tensor
