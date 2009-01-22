// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include <cassert>
#include "matrix_permute.cc"

namespace tensor {

  template<typename n> inline
  const Tensor<n> do_transpose(const Tensor<n> &a)
  {
    assert(a.rank() == 2);
    index rows = a.rows();
    index cols = a.columns();
    Tensor<n> b(cols, rows);
    permute_12(b, a, rows, cols, 1);
    return b;
  }

} // namespace tensor
