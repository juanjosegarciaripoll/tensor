// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include "matrix_trace.cc"

namespace tensor {

  double trace(const RTensor &a)
  {
    return do_trace(a);
  }

} // namespace tensor
