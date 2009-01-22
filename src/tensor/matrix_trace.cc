// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include <tensor/tensor.h>

namespace tensor {

  template<typename n>
  const n do_trace(const Tensor<n> &t)
  {
    n output = number_zero<n>();
    const index r = t.rows();
    const index c = t.columns();
    typename Tensor<n>::const_iterator it = t.begin();
    for (index j = std::min(r,c); j--; it += (r+1)) {
      output += *it;
    }
    return output;
  }

} // namespace tensor
