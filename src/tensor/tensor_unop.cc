// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include <cmath>
#include <tensor/tensor.h>

namespace tensor {

  TYPE2 OPERATOR1(const TYPE1 &t) {
    TYPE2 output(t.dimensions());
    TYPE1::const_iterator src = t.begin();
    TYPE2::iterator dest = output.begin();
    for (; src != t.end(); src++, dest++) {
      *dest = OPERATOR2(*src);
    }
    return output;
  }

} // namespace tensor
