// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include <tensor/tensor.h>

namespace tensor {

  const Tensor<cdouble> fold(const Tensor<double> &a, int ndx1,
                             const Tensor<cdouble> &b, int ndx2)
  {
    return fold(to_complex(a), ndx1, b, ndx2);
  }

  const Tensor<cdouble> mmult(const Tensor<double> &m1, const Tensor<cdouble> &m2)
  {
    return fold(to_complex(m1), -1, m2, 0);
  }

  const Tensor<cdouble> fold(const Tensor<cdouble> &a, int ndx1,
                             const Tensor<double> &b, int ndx2)
  {
    return fold(a, ndx1, to_complex(b), ndx2);
  }

  const Tensor<cdouble> mmult(const Tensor<cdouble> &m1, const Tensor<double> &m2)
  {
    return fold(m1, -1, to_complex(m2), 0);
  }

} // namespace tensor
