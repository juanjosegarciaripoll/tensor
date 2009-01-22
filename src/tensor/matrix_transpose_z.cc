// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include "matrix_transpose.cc"

namespace tensor {

  const CTensor transpose(const CTensor &a)
  {
    return do_transpose(a);
  }

} // namespace tensor
