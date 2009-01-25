// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#define TENSOR_LOAD_IMPL
#include <tensor/tensor.h>

namespace tensor {

  const CTensor to_complex(const RTensor &r)
  {
    CTensor output(r.dimensions());
    RTensor::const_iterator ir = r.begin();
    for (CTensor::iterator io = output.begin(); io != output.end(); ++io, ++ir) {
      *io = to_complex(*ir);
    }
    return output;
  }

  const CTensor to_complex(const RTensor &r, const RTensor &i)
  {
    CTensor output(r.dimensions());
    RTensor::const_iterator ir = r.begin();
    RTensor::const_iterator ii = i.begin();
    for (CTensor::iterator io = output.begin(); io != output.end(); ++io, ++ir, +ii) {
      *io = to_complex(*ir,*ii);
    }
    return output;
  }

} // namespace tensor
