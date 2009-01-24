// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include "matrix_adjoint.cc"

namespace tensor {

  const CTensor adjoint(const CTensor &a)
  {
    return do_adjoint(a);
  }

} // namespace tensor
