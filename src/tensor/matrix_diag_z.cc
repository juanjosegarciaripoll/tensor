// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include "matrix_diag.cc"

namespace tensor {

  const CTensor diag(const CTensor &a, int which, int rows, int cols)
  {
    return do_diag(a, which, rows, cols);
  }

  const CTensor diag(const CTensor &a, int which)
  {
    index n = a.size() + std::abs(which);
    return diag(a, which, n, n);
  }

} // namespace tensor
