// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include <cassert>
#include <tensor/tensor.h>

namespace tensor {

  template<typename n>
  const Tensor<n> do_diag(const Tensor<n> &a, int which, int rows, int cols)
  {
    Tensor<n> output(rows, cols);
    output.fill_with_zeros();
    index r0, c0;
    if (which < 0) {
      r0 = -which;
      c0 = 0;
    } else {
      r0 = 0;
      c0 = which;
    }
    index l = std::min<index>(rows - r0, cols - c0);
    if (l < 0) {
      std::cerr << "In diag(a,which,...) the value of WHICH exceeds the size of the matrix"
                << std::endl;
      abort();
    }
    if (l != a.size()) {
      std::cerr << "In diag(a,...) the vector A has too few/many elements."
                << std::endl;
      abort();
    }
    for (size_t i = 0; i < (size_t)l; i++) {
	output.at(r0+i,c0+i) = a[i];
    }
    return output;
  }

} // namespace tensor
