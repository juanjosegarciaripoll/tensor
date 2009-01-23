// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include "tensor_permute.cc"

namespace tensor {

  const CTensor permute(const CTensor &a, index i1, index i2)
  {
    return do_permute(a, i1, i2);
  }

} // namespace tensor
