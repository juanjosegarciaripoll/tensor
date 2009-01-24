// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include <cassert>

namespace tensor {

  template<typename n> inline
  const Tensor<n> do_adjoint(const Tensor<n> &a)
  {
    assert(a.rank() == 2);
    index rows = a.rows();
    index cols = a.columns();
    Tensor<n> b(cols, rows);
    if (cols && rows) {
      typename Tensor<n>::const_iterator ij_a = a.begin();
      typename Tensor<n>::iterator j_b = b.begin();
      for (index j = cols; j--; j_b++) {
        typename Tensor<n>::iterator ji_b = j_b;
        for (index i = rows; i--; ij_a++, ji_b += cols) {
          //assert(ijk_a >= a.begin() && ijk_a < a.end());
          //assert(ijk_b >= b.begin() && ijk_b < b.end());
          *ji_b = conj(*ijk_a);
        }
      }
    }
    return b;
  }

} // namespace tensor
