// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include "tensor_foldin.cc"

namespace tensor {

  /**Contraction of two tensors with complex conjugation. Similar to the fold()
     routine, but the indices of the first tensor are inserted in the output.
     In other words the code \c
     C=foldc(A,1,B,0) performs
     \f[
     C_{i_3i_0i_1i_4} = \sum_j A_{i_0ji_1}^\star B_{ji_3i_4}
     \f]

     \ingroup Tensors
  */
  const Tensor<cdouble>
  foldin(const Tensor<cdouble> &a, int _ndx1, const Tensor<cdouble> &b, int _ndx2)
  {
    Tensor<cdouble> output;
    do_foldin_into(output, a, _ndx1, b, _ndx2);
    return output;
  }

  /**Similar to foldin(), but the output has been preallocated.

     \ingroup Tensors
  */
  void
  foldin_into(Tensor<cdouble> &output, const Tensor<cdouble> &a, int _ndx1,
              const Tensor<cdouble> &b, int _ndx2)
  {
    do_foldin_into(output, a, _ndx1, b, _ndx2);
  }

} // namespace tensor
