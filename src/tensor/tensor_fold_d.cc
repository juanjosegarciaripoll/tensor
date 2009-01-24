// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include "tensor_fold.cc"

namespace tensor {

  /**Contraction of two tensors. A contraction is a generalization of the matrix
     product that we all know. The code \c C=fold(A,1,B,0) contracts tensors A
     and B, according to the second and first index as given by the formula
     \f[
     C_{i_0i_1i_3i_4} = \sum_j A_{i_0ji_1} B_{ji_3i_4}
     \f]

     There exist variants for these functions for contracting real with real,
     complex with complex, and real with complex tensors. Depending on the
     choice of indices, NDX1 and NDX2, the multiplication will be faster or
     slower. For instance, at least fast products are achieved via the ATLAS
     library when NDX=0 or 1.

     \ingroup Tensors
  */
  const Tensor<double> fold(const Tensor<double> &a, int ndx1,
                            const Tensor<double> &b, int ndx2)
  {
    Tensor<double> output;
    do_fold<double, false>(output, a, ndx1, b, ndx2);
    return output;
  }

  void fold_into(Tensor<double> &c, const Tensor<double> &a, int ndx1,
                 const Tensor<double> &b, int ndx2)
  {
    do_fold<double, false>(c, a, ndx1, b, ndx2);
  }

  const Tensor<double> mmult(const Tensor<double> &m1, const Tensor<double> &m2)
  {
    return fold(m1, -1, m2, 0);
  }

  void mmult_into(Tensor<double> &c, const Tensor<double> &m1, const Tensor<double> &m2)
  {
    fold_into(c, m1, -1, m2, 0);
  }

} // namespace tensor
